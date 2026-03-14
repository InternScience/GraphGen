import re
from typing import Any, Dict, List

TITLE_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+"),
    re.compile(r"^\d+(?:\.\d+)*\s+.+"),
    re.compile(r"^(?:第[一二三四五六七八九十百千万\d]+[章节篇节])\s*.+"),
]


def infer_title_level(title: str) -> int:
    stripped = (title or "").strip()
    if not stripped:
        return 1

    if stripped.startswith("#"):
        return min(6, len(stripped) - len(stripped.lstrip("#")))

    numeric = re.match(r"^(\d+(?:\.\d+)*)\s+", stripped)
    if numeric:
        return min(6, numeric.group(1).count(".") + 1)

    zh_num = re.match(r"^第([一二三四五六七八九十百千万\d]+)([章节篇节])", stripped)
    if zh_num:
        return 1 if zh_num.group(2) in {"章", "篇"} else 2

    return 1


def is_title_line(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    return any(pattern.match(line) for pattern in TITLE_PATTERNS)


def compact_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def merge_metadata(doc: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(doc.get("metadata", {}))
    base.update(extra)
    return base


def normalize_components(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = doc.get("content", "")
    lines = [ln.strip() for ln in str(content).splitlines()]
    components: List[Dict[str, Any]] = []

    current_title = "Document"
    current_buffer: List[str] = []

    for line in lines:
        if not line:
            continue
        if is_title_line(line):
            if current_buffer:
                components.append(
                    {
                        "type": "text",
                        "title": current_title,
                        "content": compact_text("\n".join(current_buffer)),
                        "title_level": infer_title_level(current_title),
                    }
                )
                current_buffer = []
            current_title = line
            continue
        current_buffer.append(line)

    if current_buffer:
        components.append(
            {
                "type": "text",
                "title": current_title,
                "content": compact_text("\n".join(current_buffer)),
                "title_level": infer_title_level(current_title),
            }
        )

    if not components and content:
        components.append(
            {
                "type": doc.get("type", "text"),
                "title": "Document",
                "content": compact_text(str(content)),
                "title_level": 1,
            }
        )

    return components
