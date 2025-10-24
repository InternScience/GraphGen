# multi_omics_search.py
import logging
import re
import time
from typing import Dict, Optional

import requests
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from requests import Session, adapters
from urllib3.util.retry import Retry

# ---------- 底层工具 ----------
_SESSION: Optional[Session] = None


def _get_session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = Session()
        retry = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
        )
        _SESSION.mount("https://", adapters.HTTPAdapter(max_retries=retry))
        _SESSION.headers.update({"User-Agent": "MultiOmicsQuery/1.0"})
    return _SESSION


# ---------- 数据抓取 ----------
def _fetch_uniprot(entry: str) -> Optional[Dict]:
    entry = entry.strip()
    if re.fullmatch(
        r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", entry
    ):
        url = f"https://www.uniprot.org/uniprot/{entry}.json"
        r = _get_session().get(url, timeout=15)
        if r.ok:
            return _parse_uniprot(r.json())
    # 模糊搜索
    kw = entry.upper().replace("-", "")
    if kw == "INTERLEUKIN6":
        kw = "IL6"
    r = _get_session().get(
        "https://www.uniprot.org/uniprot/",
        params={
            "query": f"gene:{kw} OR name:{kw} OR {kw}",
            "format": "json",
            "limit": 1,
        },
        timeout=15,
    )
    if not r.ok or not r.json().get("results"):
        return None
    acc = r.json()["results"][0]["primaryAccession"]
    return _fetch_uniprot(acc)  # 递归拿详情


def _parse_uniprot(data: dict) -> dict:
    return {
        "molecule_type": "protein",
        "database": "UniProt",
        "id": data["primaryAccession"],
        "entry_name": data["uniProtkbId"],
        "gene_names": [
            g["geneName"]["value"] for g in data.get("genes", []) if "geneName" in g
        ],
        "protein_name": (
            data["proteinDescription"]["recommendedName"]["fullName"]["value"]
            if "recommendedName" in data["proteinDescription"]
            else data["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
        ),
        "organism": data["organism"]["scientificName"],
        "sequence": data["sequence"]["value"],
        "function": " | ".join(
            [
                c["texts"][0]["value"]
                for c in data.get("comments", [])
                if c["commentType"] == "FUNCTION"
            ]
        ),
        "url": f"https://www.uniprot.org/uniprot/{data['primaryAccession']}",
    }


def _fetch_ncbi_gene(gene_id: str) -> Optional[Dict]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "gene", "id": gene_id, "retmode": "json"}
    r = requests.get(url, params=params, timeout=15)
    if not r.ok:
        return None
    data = r.json()["result"][gene_id]
    return {
        "molecule_type": "gene",
        "database": "NCBI Gene",
        "id": gene_id,
        "symbol": data["name"],
        "description": data["description"],
        "organism": data["organism"]["scientificname"],
        "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
    }


def _blast(fasta: str, mol_type: str) -> Optional[Dict]:
    program = "blastp" if mol_type == "protein" else "blastn"
    try:
        result = NCBIWWW.qblast(program, "nr", fasta, hitlist_size=1, format_type="XML")
        rec = next(NCBIXML.parse(result))
        if not rec.alignments:
            return None
        best = rec.alignments[0]
        hit_id = best.hit_id.split("|")[3] if "|" in best.hit_id else best.hit_id
        return {
            "molecule_type": mol_type,
            "database": "NCBI BLAST",
            "hit_id": hit_id,
            "hit_title": best.hit_def,
            "hit_score": best.hsps[0].score,
            "hit_evalue": best.hsps[0].expect,
            "url": f"https://www.ncbi.nlm.nih.gov/protein/{hit_id}",
        }
    except Exception as e:
        logging.warning("BLAST 失败: %s", e)
        return None


# ---------- 唯一对外接口 ----------
def search(entry: str) -> dict:
    """
    万能入口：
      - UniProt AC  → UniProt 记录
      - 纯数字      → NCBI Gene
      - FASTA       → 自动判断蛋白/核酸并 BLAST
      - 其余        → 先当蛋白名搜 UniProt
    返回统一字典；找不到时 error 字段给出原因。
    """
    entry = entry.strip()
    if not entry:
        return {"input": entry, "error": "empty query"}

    # 1. FASTA？
    if entry.startswith(">"):
        record = SeqIO.read(entry.splitlines(), "fasta")
        mol_type = (
            "protein"
            if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in str(record.seq).upper())
            else "dna"
        )
        blast_res = _blast(entry, mol_type)
        if blast_res is None:
            return {"input": entry, "error": "BLAST 无显著匹配"}
        return {"input": entry, "blast": blast_res}

    # 2. UniProt AC？
    if re.fullmatch(
        r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", entry
    ):
        uni = _fetch_uniprot(entry)
        if uni is None:
            return {"input": entry, "error": "UniProt 未找到"}
        return {"input": entry, "uniprot": uni}

    # 3. NCBI Gene ID？
    if entry.isdigit():
        gene = _fetch_ncbi_gene(entry)
        if gene is None:
            return {"input": entry, "error": "NCBI Gene 未找到"}
        return {"input": entry, "gene": gene}

    # 4. 默认按名称搜 UniProt
    uni = _fetch_uniprot(entry)
    if uni:
        return {"input": entry, "uniprot": uni}
    return {"input": entry, "error": "未找到匹配记录"}


# ---------- 使用示例 ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
    print(search("P69905"))  # UniProt AC
    print(search("7157"))  # NCBI Gene ID
    print(search(">seq\nMAAAAA"))  # FASTA
    print(search("interleukin-6"))  # 名称
