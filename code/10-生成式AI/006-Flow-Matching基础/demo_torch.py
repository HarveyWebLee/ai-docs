"""
一维 Conditional Flow Matching：常数速度场 + Euler 积分（PyTorch）。

对应文档：docs/10-生成式AI/006-Flow-Matching基础.md

运行：python code/10-生成式AI/006-Flow-Matching基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def euler_integrate(x0: torch.Tensor, velocity: torch.Tensor, steps: int) -> torch.Tensor:
    dt = 1.0 / steps
    x = x0.clone()
    for _ in range(steps):
        x = x + velocity * dt
    return x


def main() -> None:
    seed_all(42)
    dev = device()

    x0 = torch.tensor(-1.0, device=dev)
    x1 = torch.tensor(2.0, device=dev)
    velocity = x1 - x0  # 线性 CFM 真速度 u = x1 - x0

    print("一维 Flow Matching（PyTorch Euler 积分）")
    print(f"x0={x0.item()}, x1={x1.item()}, u={velocity.item()}\n")
    print(f"{'步数 N':>8} | {'x(1)':>10} | {'误差':>10}")
    print("-" * 34)

    for n in [1, 2, 4, 8, 16]:
        x_final = euler_integrate(x0, velocity, n)
        err = abs(x_final.item() - x1.item())
        print(f"{n:>8} | {x_final.item():>10.4f} | {err:>10.6f}")

    print(f"\nN=1 时一步从 {x0.item()} 到 {x1.item()}（与标准库 demo 一致）")


if __name__ == "__main__":
    main()
