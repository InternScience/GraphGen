import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from graphgen.bases.base_reader import BaseReader
from graphgen.models import TxtReader
from graphgen.utils import logger


class PdfReader(BaseReader):
    """
    PDF files are converted using MinerU, see [MinerU](https://github.com/opendatalab/MinerU).
    After conversion, the generated markdown file is read using TxtReader and pictures can be used for VQA tasks.
    """

    def __init__(
        self,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        method: str = "auto",  # auto | txt | ocr
        lang: Optional[str] = None,  # ch / en / ja / ...
        backend: Optional[
            str
        ] = None,  # pipeline | vlm-transformers | vlm-sglang-engine | vlm-sglang-client
        device: Optional[str] = None,  # cpu | cuda | cuda:0 | npu | mps
        source: Optional[str] = None,  # huggingface | modelscope | local
        vlm_url: Optional[str] = None,  # 当 backend=vlm-sglang-client 时必填
        start_page: Optional[int] = None,  # 0-based
        end_page: Optional[int] = None,  # 0-based， inclusive
        formula: bool = True,
        table: bool = True,
        **other_mineru_kwargs: Any,
    ):
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir else None

        self._default_kwargs: Dict[str, Any] = {
            "method": method,
            "lang": lang,
            "backend": backend,
            "device": device,
            "source": source,
            "vlm_url": vlm_url,
            "start_page": start_page,
            "end_page": end_page,
            "formula": formula,
            "table": table,
            **other_mineru_kwargs,
        }
        self._default_kwargs = {
            k: v for k, v in self._default_kwargs.items() if v is not None
        }

        self.parser = MinerUParser()
        self.txt_reader = TxtReader()

    def read(self, file_path: str, **override) -> List[Dict[str, Any]]:
        """
        file_path
        **override: override MinerU parameters
        """
        pdf_path = Path(file_path).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)

        kwargs = {**self._default_kwargs, **override}

        self._call_mineru(pdf_path, kwargs)

        md_file = self._locate_md(pdf_path, kwargs)
        if md_file is None:
            logger.warning(
                "Cannot locate generated markdown file for PDF: %s", pdf_path
            )
            return []

        return self.txt_reader.read(str(md_file))

    def _call_mineru(
        self, pdf_path: Path, kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        output_dir: Optional[str] = None
        if self.output_dir:
            output_dir = str(self.output_dir)

        return self.parser.parse_pdf(pdf_path, output_dir=output_dir, **kwargs)

    def _locate_md(self, pdf_path: Path, kwargs: Dict[str, Any]) -> Optional[Path]:
        out_dir = (
            Path(self.output_dir) if self.output_dir else Path(tempfile.gettempdir())
        )
        method = kwargs.get("method", "auto")
        backend = kwargs.get("backend", "")
        if backend.startswith("vlm-"):
            method = "vlm"

        candidate = out_dir / pdf_path.stem / method / f"{pdf_path.stem}.md"
        if candidate.exists():
            return candidate
        candidate2 = out_dir / f"{pdf_path.stem}.md"
        if candidate2.exists():
            return candidate2
        return None


class MinerUParser:
    def __init__(self) -> None:
        self._check_bin()

    @staticmethod
    def parse_pdf(
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        method: str = "auto",
        device: str = "cpu",
        **kw: Any,
    ) -> List[Dict[str, Any]]:
        pdf = Path(pdf_path).expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(pdf)

        out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="mu_"))
        out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mineru",
            "-p",
            str(pdf),
            "-o",
            str(out),
            "-m",
            method,
            "-d",
            device,
        ]
        for k, v in kw.items():
            if v is None:
                continue
            if isinstance(v, bool):
                cmd += [f"--{k}", str(v).lower()]
            else:
                cmd += [f"--{k}", str(v)]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,  # catch later
        )
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU failed: {proc.stderr or proc.stdout}")

        json_file = out / f"{pdf.stem}_content_list.json"
        if not json_file.exists():
            json_file = out / pdf.stem / method / f"{pdf.stem}_content_list.json"

        if json_file.exists():
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
                base = json_file.parent
                for item in data:
                    for key in ("img_path", "table_img_path", "equation_img_path"):
                        if item.get(key):
                            item[key] = str((base / item[key]).resolve())
                return data
        return []

    @staticmethod
    def _check_bin() -> None:
        try:
            subprocess.run(
                ["mineru", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "MinerU is not installed or not found in PATH. Please install it from pip: \n"
                "pip install -U 'mineru[core]'"
            ) from exc


if __name__ == "__main__":
    # Simple test
    sample_pdf = "resources/input_examples/pdf_demo.pdf"
    parser = MinerUParser()
    blocks = parser.parse_pdf(
        sample_pdf, device="cpu", method="auto", output_dir="cache"
    )
    print(f"Parsed {len(blocks)} blocks from {sample_pdf}")
