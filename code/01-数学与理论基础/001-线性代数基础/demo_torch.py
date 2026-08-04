"""
用 PyTorch 复现文档 2×2 矩阵特征值手算案例，并演示 SVD。

对应文档：docs/01-数学与理论基础/001-线性代数基础.md

运行：python code/01-数学与理论基础/001-线性代数基础/demo_torch.py
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

    a = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device=dev)

    # 对称矩阵可用 eigh（实特征值）
    evals, evecs = torch.linalg.eigh(a)
    u, s, vh = torch.linalg.svd(a)

    print("2×2 矩阵 A = [[2,1],[1,2]] 的特征分解（PyTorch）")
    print(f"特征值 λ（升序）: {evals.tolist()}  → 文档 λ₁=1, λ₂=3")
    print(f"特征向量列（对应 λ）:\n{evecs}\n")

    print("SVD: A = U Σ V^T")
    print(f"奇异值 Σ 对角: {s.tolist()}")
    recon = u @ torch.diag(s) @ vh
    err = (recon - a).abs().max().item()
    print(f"重构误差 max|UΣV^T - A| = {err:.2e}（应≈0）")


if __name__ == "__main__":
    main()
