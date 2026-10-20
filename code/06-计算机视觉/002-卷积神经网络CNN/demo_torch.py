"""
微型 CNN 在 8×8 二值图案上二分类（横条 vs 竖条），少量 epoch 快速收敛。

对应文档：docs/06-计算机视觉/002-卷积神经网络CNN.md

运行：python code/06-计算机视觉/002-卷积神经网络CNN/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """卷积 → 池化 → 全连接，对应文档中的 CNN 流水线。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_patterns(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """生成横条(0) / 竖条(1) 二值图案。"""
    xs, ys = [], []
    for i in range(n):
        img = torch.zeros(1, 8, 8)
        if i % 2 == 0:
            img[:, 3:5, :] = 1.0  # 横条
            label = 0
        else:
            img[:, :, 3:5] = 1.0  # 竖条
            label = 1
        xs.append(img)
        ys.append(label)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def main() -> None:
    seed_all(42)
    dev = device()

    x, y = make_patterns(32)
    x, y = x.to(dev), y.to(dev)

    model = TinyCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    print("微型 CNN：8×8 二值图案，类别 0=横条 / 1=竖条")
    for epoch in range(8):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean().item()
        print(f"epoch {epoch + 1:2d} | loss={loss.item():.4f} | acc={acc:.2%}")

    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(1)
    print(f"\n最终预测全对：{(pred == y).all().item()}")


if __name__ == "__main__":
    main()
