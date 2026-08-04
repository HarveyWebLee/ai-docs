"""
表格型 Q-learning 单步更新：Q=10 → 10.55（与文档案例一致）。

对应文档：docs/07-强化学习/003-Q学习与时序差分.md

运行：python code/07-强化学习/003-Q学习与时序差分/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def q_update(
    q: torch.Tensor,
    s: int,
    a: int,
    reward: float,
    s_next: int,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Q(s,a) ← Q(s,a) + α [R + γ max Q(s',·) - Q(s,a)]"""
    td_target = reward + gamma * q[s_next].max()
    td_error = td_target - q[s, a]
    q[s, a] = q[s, a] + alpha * td_error
    return q


def main() -> None:
    seed_all(42)
    dev = device()

    # 小网格：2 状态 × 2 动作
    q_table = torch.zeros(2, 2, device=dev)
    q_table[0, 0] = 10.0
    q_table[1, 0] = 12.0
    q_table[1, 1] = 15.0  # max Q(s',a') = 15

    s, a, s_next = 0, 0, 1
    reward, gamma, alpha = 2.0, 0.9, 0.1

    td_target = reward + gamma * q_table[s_next].max()
    td_error = td_target - q_table[s, a]
    old_q = q_table[s, a].item()

    q_table = q_update(q_table, s, a, reward, s_next, gamma, alpha)
    new_q = q_table[s, a].item()

    print("Q-learning 单步更新（文档案例）")
    print(f"Q(s,a) 初值     = {old_q:.2f}")
    print(f"R={reward}, max Q(s',a')={q_table[s_next].max().item():.0f}, γ={gamma}, α={alpha}")
    print(f"TD 目标         = {reward} + {gamma}×15 = {td_target.item():.2f}")
    print(f"TD 误差 δ       = {td_target.item():.2f} - {old_q:.2f} = {td_error.item():.2f}")
    print(f"Q(s,a) 更新后   = {old_q:.2f} + {alpha}×{td_error.item():.2f} = {new_q:.2f}")
    print(f"\n验证 Q: 10 → 10.55: {abs(new_q - 10.55) < 1e-6}")


if __name__ == "__main__":
    main()
