"""
REINFORCE 在两臂 bandit 上：高回报臂的概率经梯度上升后增大。

对应文档：docs/07-强化学习/004-策略梯度与深度强化学习.md

运行：python code/07-强化学习/004-策略梯度与深度强化学习/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.nn.functional as F


class BanditPolicy(nn.Module):
    """输出两动作 logits 的简单策略网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)


def main() -> None:
    seed_all(42)
    dev = device()

    policy = BanditPolicy().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=0.1)

    # 臂 1 回报高、臂 0 回报低
    rewards = [0.0, 1.0]

    probs_before = policy().detach().cpu().tolist()
    print("REINFORCE：2-arm bandit（臂1 高回报）")
    print(f"更新前概率 [臂0, 臂1]: {[f'{p:.3f}' for p in probs_before]}")

    for step in range(30):
        probs = policy()
        action = torch.multinomial(probs, 1).item()
        G = rewards[action]
        log_prob = torch.log(probs[action] + 1e-8)
        loss = -log_prob * G
        opt.zero_grad()
        loss.backward()
        opt.step()

    probs_after = policy().detach().cpu().tolist()
    print(f"更新后概率 [臂0, 臂1]: {[f'{p:.3f}' for p in probs_after]}")
    increased = probs_after[1] > probs_before[1]
    print(f"\n高回报臂(1)概率上升: {increased} ({probs_before[1]:.3f} → {probs_after[1]:.3f})")


if __name__ == "__main__":
    main()
