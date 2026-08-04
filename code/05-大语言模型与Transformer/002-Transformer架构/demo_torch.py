"""
单层 TransformerEncoderLayer 前向传播，打印输出张量形状。

对应文档：docs/05-大语言模型与Transformer/002-Transformer架构.md

运行：python code/05-大语言模型与Transformer/002-Transformer架构/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


def main() -> None:
    seed_all(42)
    dev = device()

    batch, seq_len, d_model = 2, 5, 32
    nhead = 4

    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=64,
        batch_first=True,
    ).to(dev)

    x = torch.randn(batch, seq_len, d_model, device=dev)
    out = layer(x)

    print("TransformerEncoderLayer 单次前向")
    print(f"输入形状：{tuple(x.shape)}  (batch, seq_len, d_model)")
    print(f"输出形状：{tuple(out.shape)}")
    print(f"验证：输出与输入形状一致 → {out.shape == x.shape}")
    print(f"参数量：{sum(p.numel() for p in layer.parameters()):,}")


if __name__ == "__main__":
    main()
