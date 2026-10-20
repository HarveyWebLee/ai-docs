"""
浅层 MLP 与「轴对齐规则」对比：表格数据二分类（PyTorch）。

对应文档：docs/02-机器学习基础/005-决策树与集成方法.md
决策树擅长轴对齐边界；本 demo 用 2 特征 XOR-like 数据展示 MLP 可学非线性。

运行：python code/02-机器学习基础/005-决策树与集成方法/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim


def make_xor_2d(n: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 2, device=dev)
    y = ((x[:, 0] > 0.5) ^ (x[:, 1] > 0.5)).float()
    return x, y


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    seed_all(42)
    dev = device()
    x, y = make_xor_2d(512, dev)

    model = MLP().to(dev)
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = ((torch.sigmoid(model(x)) >= 0.5).float() == y).float().mean().item()

    print("表格二分类：2 特征 XOR 型边界（PyTorch MLP）")
    print(f"训练准确率 ≈ {acc:.1%}")
    print("对比：单棵决策树可用轴对齐规则；集成/神经网络可拟合更复杂边界。")
    print("标准库决策树 demo：code/02-机器学习基础/005-决策树与集成方法/decision_tree_demo.py")


if __name__ == "__main__":
    main()
