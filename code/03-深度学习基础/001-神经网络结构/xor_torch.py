"""
用 PyTorch 训练两层 ReLU MLP 解决 XOR，验证 4/4 真值表正确。

对应文档：docs/03-深度学习基础/001-神经网络结构.md
线性不可分问题需隐层非线性；本脚本从零训练权重（非手工给定）。

运行：python code/03-深度学习基础/001-神经网络结构/xor_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim


class XorMLP(nn.Module):
    """2 输入 → 隐层 ReLU → 1 输出（Sigmoid 二分类）。"""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    seed_all(42)
    dev = device()

    # XOR 真值表
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=dev)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=dev)

    model = XorMLP(hidden=8).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        preds = (probs >= 0.5).float()
        correct = (preds == y).sum().item()

    print("XOR 两层 MLP（ReLU + BCE，PyTorch 训练）")
    print(f"最终 loss={loss.item():.4f}\n")
    print("x1 x2 | 预测 | 期望 | 一致")
    print("------+------+------+-----")
    labels = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, (a, b) in enumerate(labels):
        p = preds[i].item()
        t = y[i].item()
        mark = "OK" if p == t else "FAIL"
        print(f" {a}  {b} |  {p:.0f}  |  {t:.0f}  | {mark}")

    print(f"\n准确率：{correct}/4")
    print("结论：", "全部正确，隐层非线性可学 XOR。" if correct == 4 else "未完全收敛，可增大 epoch。")


if __name__ == "__main__":
    main()
