"""
折扣因子 γ 如何改变「立刻 +10 vs 下一步 +100」的最优选择（PyTorch）。

对应文档：docs/07-强化学习/001-马尔可夫决策过程MDP.md

运行：python code/07-强化学习/001-马尔可夫决策过程MDP/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def discounted_value(reward: float, steps: int, gamma: float) -> float:
    """一步延迟的奖励折算到当前的现值。"""
    return (gamma ** steps) * reward


def main() -> None:
    seed_all(42)
    dev = device()

    immediate = 10.0
    delayed = 100.0
    gammas = torch.tensor([0.05, 0.5, 0.9], device=dev)

    print("MDP 案例：立刻 +10 vs 忍耐一步后 +100")
    print(f"{'γ':>6} | {'延迟奖励现值 γ×100':>18} | 更优选择")
    print("-" * 45)

    for g in gammas:
        g_val = g.item()
        present = discounted_value(delayed, steps=1, gamma=g_val)
        better = "忍耐 (+100)" if present > immediate else "立刻 (+10)"
        print(f"{g_val:>6.2f} | {present:>18.2f} | {better}")

    print("\n验证：γ=0.5 时 50>10 应忍耐；γ=0.05 时 5<10 应立刻（与文档一致）")


if __name__ == "__main__":
    main()
