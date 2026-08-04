"""
正态采样、经验均值/方差与 Softmax 概率分布（PyTorch）。

对应文档：docs/01-数学与理论基础/002-概率与统计基础.md

运行：python code/01-数学与理论基础/002-概率与统计基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn.functional as F


def main() -> None:
    seed_all(42)
    dev = device()

    n = 10_000
    mu, sigma = 3.0, 1.5
    samples = torch.normal(mean=mu, std=sigma, size=(n,), device=dev)

    print(f"从 N(μ={mu}, σ={sigma}) 采样 {n} 个点")
    print(f"样本均值 ≈ {samples.mean().item():.3f}（理论 μ={mu}）")
    print(f"样本标准差 ≈ {samples.std(unbiased=True).item():.3f}（理论 σ={sigma}）\n")

    logits = torch.tensor([2.0, 1.0, 0.1], device=dev)
    probs = F.softmax(logits, dim=0)
    print("Softmax 把「打分」变成「概率和为 1 的分布」")
    print(f"logits = {logits.tolist()}")
    print(f"probs  = {[round(p, 4) for p in probs.tolist()]}")
    print(f"sum(probs) = {probs.sum().item():.6f}")


if __name__ == "__main__":
    main()
