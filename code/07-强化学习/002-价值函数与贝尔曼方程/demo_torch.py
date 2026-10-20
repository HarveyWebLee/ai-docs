"""
贝尔曼期望备份：V(s) ← R + γ V(s')（PyTorch 张量版）。

对应文档：docs/07-强化学习/002-价值函数与贝尔曼方程.md

运行：python code/07-强化学习/002-价值函数与贝尔曼方程/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def bellman_backup(
    v: torch.Tensor,
    s: int,
    reward: float,
    s_next: int,
    gamma: float,
) -> torch.Tensor:
    """V(s) ← R + γ V(s')"""
    v_new = v.clone()
    v_new[s] = reward + gamma * v[s_next]
    return v_new


def main() -> None:
    seed_all(42)
    dev = device()

    # 3 状态网格，初值 V=0
    v = torch.zeros(3, device=dev)
    v[2] = 10.0  # 终点状态价值

    s, s_next = 0, 1
    reward, gamma = 2.0, 0.9

    v_before = v[s].item()
    v = bellman_backup(v, s, reward, s_next, gamma)
    v_after = v[s].item()

    # 再备份一步：s=1 → s=2
    v = bellman_backup(v, 1, 1.0, 2, gamma)

    print("贝尔曼备份 V(s) ← R + γ V(s')")
    print(f"状态 2 终点价值 V(2)={v[2].item():.1f}")
    print(f"从 s=0: R={reward}, s'=1, γ={gamma}")
    print(f"  V(0): {v_before:.2f} → {v_after:.2f} = {reward} + {gamma}×V(1)")
    print(f"再备份 s=1→2 后 V = {v.tolist()}")
    print("结论：价值从终点向前「倒推」传播，即动态规划/时序差分的基础。")


if __name__ == "__main__":
    main()
