import json
import re
from typing import Any, Dict, List, Tuple

TITLE_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+"),
    re.compile(r"^(?:\d+(?:\.\d+)+(?:\s+.*)?|\d+\s+.+)$"),
    re.compile(r"^(?:第[一二三四五六七八九十百千万\d]+[章节篇节])\s*.+"),
]


def infer_title_level(title: str) -> int:
    stripped = (title or "").strip()
    if not stripped:
        return 1

    markdown_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
    markdown_level = len(markdown_match.group(1)) if markdown_match else 0
    semantic_source = markdown_match.group(2).strip() if markdown_match else stripped

    numeric = re.match(r"^(\d+(?:\.\d+)*)(?:\s+.+)?$", semantic_source)
    if numeric:
        numeric_level = min(6, numeric.group(1).count(".") + 1)
        return max(markdown_level, numeric_level)

    zh_num = re.match(r"^第([一二三四五六七八九十百千万\d]+)([章节篇节])", semantic_source)
    if zh_num:
        zh_level = 1 if zh_num.group(2) in {"章", "篇"} else 2
        return max(markdown_level, zh_level)

    if markdown_level:
        return markdown_level

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


def _coerce_content(doc: Dict[str, Any]) -> str:
    content = doc.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _make_text_component(title: str, lines: List[str]) -> Dict[str, Any]:
    return {
        "type": "text",
        "title": title,
        "content": compact_text("\n".join(lines)),
        "title_level": infer_title_level(title),
    }


def _make_section_component(title: str) -> Dict[str, Any]:
    return {
        "type": "section",
        "title": title,
        "content": "",
        "title_level": infer_title_level(title),
    }


def _split_trailing_paragraph(lines: List[str]) -> Tuple[List[str], str]:
    if not lines:
        return [], ""

    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    if end == 0:
        return [], ""

    start = end - 1
    while start > 0 and lines[start - 1].strip():
        start -= 1

    paragraph = compact_text("\n".join(lines[start:end]))
    if not paragraph:
        return lines[:start], ""
    return lines[:start], paragraph


def _is_table_caption(line: str) -> bool:
    stripped = (line or "").strip()
    return bool(re.match(r"^(table|tab\.?)\s*\d+[\.:]?\s+", stripped, re.IGNORECASE))


def _is_image_line(line: str) -> bool:
    stripped = (line or "").strip()
    return bool(
        re.match(r"^!\[[^\]]*\]\([^)]+\)", stripped)
        or re.search(r"<img\b[^>]*src=['\"][^'\"]+['\"][^>]*>", stripped, re.IGNORECASE)
    )


def _extract_image_path(line: str) -> str:
    stripped = (line or "").strip()
    markdown_match = re.match(r"^!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", stripped)
    if markdown_match:
        return markdown_match.group(1)

    html_match = re.search(
        r"<img\b[^>]*src=['\"]([^'\"]+)['\"][^>]*>",
        stripped,
        re.IGNORECASE,
    )
    if html_match:
        return html_match.group(1)
    return ""


def _normalize_mm_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(metadata)
    caption = normalized.get("table_caption")
    if isinstance(caption, str):
        normalized["table_caption"] = [caption] if caption else []
    elif caption is None:
        normalized["table_caption"] = []

    image_caption = normalized.get("image_caption")
    if isinstance(image_caption, str):
        normalized["image_caption"] = [image_caption] if image_caption else []
    elif image_caption is None:
        normalized["image_caption"] = []

    notes = normalized.get("note_text")
    if isinstance(notes, list):
        normalized["note_text"] = compact_text("\n".join(str(it) for it in notes if it))
    elif notes is None:
        normalized["note_text"] = ""
    else:
        normalized["note_text"] = compact_text(str(notes))

    return normalized


def _build_table_content(caption_lines: List[str], table_body: str) -> str:
    parts = []
    caption_text = compact_text("\n".join(caption_lines))
    if caption_text:
        parts.append(f"[Table Caption]\n{caption_text}")
    if table_body:
        parts.append(f"[Table Body]\n{table_body}")
    return "\n\n".join(parts).strip()


def _build_image_content(caption_lines: List[str], note_text: str) -> str:
    parts = []
    caption_text = compact_text("\n".join(caption_lines))
    if caption_text:
        parts.append(caption_text)
    if note_text:
        parts.append(f"[Notes]\n{note_text}")
    return "\n\n".join(parts).strip()


def _consume_trailing_image_lines(lines: List[str], start_idx: int) -> Tuple[int, List[str], List[str]]:
    idx = start_idx
    caption_lines: List[str] = []
    note_lines: List[str] = []
    blank_seen = False

    while idx < len(lines):
        raw_line = lines[idx]
        stripped = raw_line.strip()

        if not stripped:
            blank_seen = True
            idx += 1
            continue

        if is_title_line(stripped) or _is_image_line(stripped) or stripped.lower().startswith("<table"):
            break

        if blank_seen and caption_lines:
            break

        if re.match(r"^(note|notes?)\s*[:\-]", stripped, re.IGNORECASE):
            note_lines.append(stripped)
        else:
            caption_lines.append(stripped)
        idx += 1

    return idx, caption_lines, note_lines


def _parse_markdown_components(content: str) -> List[Dict[str, Any]]:
    lines = str(content).splitlines()
    components: List[Dict[str, Any]] = []
    current_title = "Document"
    current_buffer: List[str] = []
    idx = 0

    def flush_text_buffer() -> None:
        nonlocal current_buffer
        if compact_text("\n".join(current_buffer)):
            components.append(_make_text_component(current_title, current_buffer))
        current_buffer = []

    while idx < len(lines):
        raw_line = lines[idx]
        line = raw_line.strip()

        if not line:
            current_buffer.append("")
            idx += 1
            continue

        if is_title_line(line):
            flush_text_buffer()
            current_title = line
            components.append(_make_section_component(current_title))
            idx += 1
            continue

        if line.lower().startswith("<table"):
            leading_lines, caption = _split_trailing_paragraph(current_buffer)
            use_caption = caption if _is_table_caption(caption) else ""
            current_buffer = leading_lines if use_caption else current_buffer
            flush_text_buffer()

            table_lines = [raw_line]
            idx += 1
            if "</table>" not in line.lower():
                while idx < len(lines):
                    table_lines.append(lines[idx])
                    if "</table>" in lines[idx].lower():
                        idx += 1
                        break
                    idx += 1

            table_body = compact_text("\n".join(table_lines))
            caption_lines = [use_caption] if use_caption else []
            metadata = _normalize_mm_payload(
                {
                    "table_body": table_body,
                    "table_caption": caption_lines,
                }
            )
            components.append(
                {
                    "type": "table",
                    "title": current_title,
                    "content": _build_table_content(metadata["table_caption"], table_body),
                    "title_level": infer_title_level(current_title),
                    "metadata": metadata,
                }
            )
            continue

        if _is_image_line(line):
            flush_text_buffer()
            img_path = _extract_image_path(line)
            idx, caption_lines, note_lines = _consume_trailing_image_lines(lines, idx + 1)
            note_text = compact_text("\n".join(note_lines))
            metadata = _normalize_mm_payload(
                {
                    "img_path": img_path,
                    "image_caption": caption_lines,
                    "note_text": note_text,
                }
            )
            components.append(
                {
                    "type": "image",
                    "title": current_title,
                    "content": _build_image_content(
                        metadata["image_caption"], metadata["note_text"]
                    ),
                    "title_level": infer_title_level(current_title),
                    "metadata": metadata,
                }
            )
            continue

        current_buffer.append(raw_line)
        idx += 1

    flush_text_buffer()
    return [
        component
        for component in components
        if component.get("type") == "section"
        or component.get("content")
        or component.get("metadata")
    ]


def normalize_components(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = _coerce_content(doc)
    components = _parse_markdown_components(content)

    if not components and content:
        components.append(
            {
                "type": doc.get("type", "text"),
                "title": "Document",
                "content": compact_text(content),
                "title_level": 1,
            }
        )

    return components
