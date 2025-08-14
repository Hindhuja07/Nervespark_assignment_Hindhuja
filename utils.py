import re
from rag_pipeline.parsers import FUNC_RE, HEAD_RE, lang_from_filename
import chardet

def is_text_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            raw = f.read(4096)
        if not raw:
            return False
        # crude binary check
        text_characters = bytearray({7,8,9,10,12,13,27} | set(range(0x20,0x100)))
        return all((c in text_characters) for c in raw[:1024])
    except Exception:
        return False

def simple_chunker(text: str, filename: str, max_lines=200):
    """
    Very small chunker:
     - split by headings
     - split by function/class regex
     - fallback to line-based chunks with overlap
    Returns list of dicts: {"page_content":..., "metadata": {"start_line", "end_line", "lang"}}
    """
    lines = text.splitlines()
    lang = lang_from_filename(filename)
    chunks = []

    # split by header lines
    header_idxs = [i for i, ln in enumerate(lines) if HEAD_RE.search(ln)]
    split_points = set([0, len(lines)])
    split_points.update(header_idxs)
    split_points = sorted(list(split_points))

    for i in range(len(split_points)-1):
        start = split_points[i]
        end = split_points[i+1]
        block = "\n".join(lines[start:end]).strip()
        if not block:
            continue
        # further split by functions if too large
        func_idxs = [m.start() for m in FUNC_RE.finditer(block)]
        if len(block.splitlines()) > max_lines and func_idxs:
            # naive: chop by line ranges
            sublines = block.splitlines()
            for s in range(0, len(sublines), max_lines):
                chunk_text = "\n".join(sublines[s:s+max_lines])
                chunks.append({
                    "page_content": chunk_text,
                    "metadata": {"start_line": start + s + 1, "end_line": start + s + len(chunk_text.splitlines()), "lang": lang}
                })
        else:
            chunks.append({
                "page_content": block,
                "metadata": {"start_line": start+1, "end_line": end, "lang": lang}
            })

    # fallback if nothing produced
    if not chunks and lines:
        for i in range(0, len(lines), max_lines):
            chunk_text = "\n".join(lines[i:i+max_lines])
            chunks.append({
                "page_content": chunk_text,
                "metadata": {"start_line": i+1, "end_line": i+len(chunk_text.splitlines()), "lang": lang}
            })

    return chunks
