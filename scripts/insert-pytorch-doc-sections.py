#!/usr/bin/env python3
"""为含 PyTorch 示例的文档插入「PyTorch 可运行示例」章节（若尚未存在）。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DOCS = ROOT / "docs"

INSTALL_BLOCK = """安装 CPU 版 PyTorch（仅需一次）：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
"""

SUMMARY_RE = re.compile(r"^## [^#\n]+、一句话总结\s*$", re.MULTILINE)
PYTORCH_SECTION_MARKER = "## PyTorch 可运行示例"


def doc_path_for_code_dir(code_dir: Path) -> Path | None:
    # code/03-深度学习基础/001-神经网络结构 -> docs/03-深度学习基础/001-神经网络结构.md
    rel = code_dir.relative_to(CODE)
    parts = rel.parts
    if len(parts) != 2:
        return None
    cat, chapter = parts
    # chapter dir name may not match md filename exactly - find by prefix
    doc_dir = DOCS / cat
    prefix = chapter.split("-")[0]  # 001
    matches = list(doc_dir.glob(f"{prefix}-*.md"))
    if not matches:
        return None
    return matches[0]


def collect_torch_scripts() -> dict[Path, list[str]]:
    grouped: dict[Path, list[str]] = {}
    for py in CODE.rglob("*.py"):
        if "_common" in py.parts:
            continue
        name = py.name
        if name == "demo_torch.py" or name.endswith("_torch.py"):
            rel = py.relative_to(ROOT).as_posix()
            grouped.setdefault(py.parent, []).append(rel)
    for k in grouped:
        grouped[k].sort()
    return grouped


def build_section(scripts: list[str]) -> str:
    run_lines = "\n".join(f"python {s}" for s in scripts)
    return f"""{PYTORCH_SECTION_MARKER}

{INSTALL_BLOCK.strip()}

运行 PyTorch 版示例：

```bash
{run_lines}
```
"""


def insert_section(content: str, section: str) -> str:
    if PYTORCH_SECTION_MARKER in content:
        return content
    m = SUMMARY_RE.search(content)
    if m:
        idx = m.start()
        return content[:idx].rstrip() + "\n\n" + section.rstrip() + "\n\n" + content[idx:].lstrip()
    return content.rstrip() + "\n\n" + section.rstrip() + "\n"


def main() -> None:
    grouped = collect_torch_scripts()
    updated = 0
    for code_dir, scripts in sorted(grouped.items(), key=lambda x: str(x[0])):
        doc = doc_path_for_code_dir(code_dir)
        if doc is None or not doc.exists():
            print(f"SKIP (no doc): {code_dir}")
            continue
        section = build_section(scripts)
        text = doc.read_text(encoding="utf-8")
        new_text = insert_section(text, section)
        if new_text != text:
            doc.write_text(new_text, encoding="utf-8")
            updated += 1
            print(f"UPDATED {doc.relative_to(ROOT)}")
        else:
            print(f"OK {doc.relative_to(ROOT)}")
    print(f"\n共更新 {updated} 篇文档")


if __name__ == "__main__":
    main()
