"""在 code/ 子目录的 demo 中导入 _common.torch_utils。"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_code_path() -> Path:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if parent.name == "code":
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("无法定位 code/ 根目录")
