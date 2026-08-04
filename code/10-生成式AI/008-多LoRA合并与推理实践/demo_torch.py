"""
多 LoRA 线性合并：W_eff = W0 + w1·B1@A1 + w2·B2@A2（PyTorch）。

对应文档：docs/10-生成式AI/008-多LoRA合并与推理实践.md

运行：python code/10-生成式AI/008-多LoRA合并与推理实践/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def main() -> None:
    seed_all(42)
    dev = device()

    w0 = torch.eye(2, device=dev)
    b1 = torch.tensor([[0.5], [0.0]], device=dev)
    a1 = torch.tensor([[0.2, 0.0]], device=dev)
    b2 = torch.tensor([[0.0], [0.5]], device=dev)
    a2 = torch.tensor([[0.0, 0.2]], device=dev)

    w1, w2 = 0.8, 0.6
    s1, s2 = 1.0, 1.0

    d1 = b1 @ a1
    d2 = b2 @ a2

    w_dyn = w0.clone()
    w_dyn += w1 * s1 * d1
    w_dyn += w2 * s2 * d2

    w_merge = w0 + w1 * s1 * d1 + w2 * s2 * d2

    diff = (w_dyn - w_merge).norm().item()

    print("多 LoRA 合并（PyTorch 矩阵乘）")
    print(f"W0 =\n{w0}\n")
    print(f"动态叠加 W_eff =\n{w_dyn}")
    print(f"离线合并 W_eff =\n{w_merge}")
    print(f"\nFrobenius 差 = {diff:.6f}（应≈0）")


if __name__ == "__main__":
    main()
