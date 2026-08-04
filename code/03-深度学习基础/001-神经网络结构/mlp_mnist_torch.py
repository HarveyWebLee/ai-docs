"""
8×8 简化「数字」二分类 MLP（PyTorch 版，0 vs 1）。

对应文档：docs/03-深度学习基础/001-神经网络结构.md
复用 mnist_like_mlp.py 的硬编码 8×8 模式与噪声策略，用 torch 训练。

运行：python code/03-深度学习基础/001-神经网络结构/mlp_mnist_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim

PATTERN_0 = [
    "01111110",
    "11000011",
    "10000001",
    "10000001",
    "10000001",
    "10000001",
    "11000011",
    "01111110",
]
PATTERN_1 = [
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
]


def pattern_to_vector(rows: list[str]) -> torch.Tensor:
    vals = [1.0 if c == "1" else 0.0 for row in rows for c in row]
    return torch.tensor(vals)


def make_dataset(n_per_class: int, flip_prob: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """每类 n 个带随机翻转噪声的 64 维样本。"""
    base0 = pattern_to_vector(PATTERN_0)
    base1 = pattern_to_vector(PATTERN_1)
    xs, ys = [], []
    for _ in range(n_per_class):
        x0 = torch.where(torch.rand(64) > flip_prob, base0, 1.0 - base0)
        x1 = torch.where(torch.rand(64) > flip_prob, base1, 1.0 - base1)
        xs.extend([x0, x1])
        ys.extend([0.0, 1.0])
    x = torch.stack(xs)
    y = torch.tensor(ys)
    perm = torch.randperm(len(y))
    return x[perm], y[perm]


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        return (preds == y).float().mean().item()


def main() -> None:
    seed_all(42)
    dev = device()

    x_train, y_train = make_dataset(n_per_class=8)
    x_test, y_test = make_dataset(n_per_class=4)
    x_train, y_train = x_train.to(dev), y_train.to(dev)
    x_test, y_test = x_test.to(dev), y_test.to(dev)

    model = TinyMLP(hidden=16).to(dev)
    opt = optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 200
    for epoch in range(epochs):
        opt.zero_grad()
        loss = criterion(model(x_train).squeeze(-1), y_train)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} | BCE = {loss.item():.4f}")

    acc = accuracy(model, x_test, y_test)
    print(f"\n数据形状：train {tuple(x_train.shape)}, test {tuple(x_test.shape)}")
    print(f"测试集准确率：{acc * 100:.1f}%")


if __name__ == "__main__":
    main()
