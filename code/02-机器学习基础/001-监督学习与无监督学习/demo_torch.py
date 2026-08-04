"""
在二维合成数据上训练极小线性分类器（监督学习二分类）。

对应文档：docs/02-机器学习基础/001-监督学习与无监督学习.md
用 torch 生成两类高斯 blob，线性层 + BCE 训练，打印损失与准确率。

运行：python code/02-机器学习基础/001-监督学习与无监督学习/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim


def make_blobs(n_per_class: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    """两类二维高斯 blob：类 0 中心 (-1,-1)，类 1 中心 (1,1)。"""
    torch.manual_seed(42)
    c0 = torch.randn(n_per_class, 2) * 0.4 + torch.tensor([-1.0, -1.0])
    c1 = torch.randn(n_per_class, 2) * 0.4 + torch.tensor([1.0, 1.0])
    x = torch.cat([c0, c1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return x, y


class LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main() -> None:
    seed_all(42)
    dev = device()

    x, y = make_blobs(n_per_class=30)
    x, y = x.to(dev), y.to(dev)

    model = LinearClassifier().to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze(-1))
        preds = (probs >= 0.5).float()
        acc = (preds == y).float().mean().item()

    print("二维 blob 线性二分类（监督学习，PyTorch）")
    print(f"数据形状 X={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"训练 {epochs} 轮后：loss={loss.item():.4f}, accuracy={acc * 100:.1f}%")
    print("（线性边界可分开两类中心相距较远的 blob）")


if __name__ == "__main__":
    main()
