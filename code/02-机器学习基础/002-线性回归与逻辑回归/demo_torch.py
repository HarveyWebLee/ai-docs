"""
线性回归（MSE）与逻辑回归（BCE）PyTorch 演示。

对应文档：docs/02-机器学习基础/002-线性回归与逻辑回归.md
(a) 文档同款一维点拟合 y≈2x+1；(b) 验证 σ(2x-4) 在 x=3 时 ≈0.881。

运行：python code/02-机器学习基础/002-线性回归与逻辑回归/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim


# 文档线性回归样本：(x, y) 近似 y = 2x + 1
REGRESSION_DATA = [
    (1.0, 3.1),
    (2.0, 4.9),
    (3.0, 7.2),
    (4.0, 8.8),
    (5.0, 11.2),
]


def demo_linear_regression(dev: torch.device) -> None:
    xs = torch.tensor([d[0] for d in REGRESSION_DATA], device=dev).unsqueeze(1)
    ys = torch.tensor([d[1] for d in REGRESSION_DATA], device=dev).unsqueeze(1)

    w = torch.tensor(0.0, device=dev, requires_grad=True)
    b = torch.tensor(0.0, device=dev, requires_grad=True)
    optimizer = optim.SGD([w, b], lr=0.05)

    for _ in range(200):
        optimizer.zero_grad()
        pred = w * xs + b
        loss = nn.functional.mse_loss(pred, ys)
        loss.backward()
        optimizer.step()

    print("(a) 线性回归 MSE")
    print(f"    样本数={len(REGRESSION_DATA)}, 最终 MSE={loss.item():.4f}")
    print(f"    拟合：y_hat = {w.item():.3f}·x + {b.item():.3f}  （真值约 2x+1）")
    x_test = 6.0
    print(f"    预测 x={x_test} → y_hat={(w.item() * x_test + b.item()):.2f}\n")


def demo_logistic_regression(dev: torch.device) -> None:
    # 文档手算：w=2, b=-4，x=3 → z=2, σ(z)≈0.881
    w = torch.tensor(2.0, device=dev)
    b = torch.tensor(-4.0, device=dev)
    x = torch.tensor(3.0, device=dev)
    z = w * x + b
    p = torch.sigmoid(z)

    print("(b) 逻辑回归 Sigmoid 验证")
    print(f"    z = 2·3 - 4 = {z.item():.1f}")
    print(f"    σ(z) = {p.item():.4f}  （文档 ≈ 0.881）")
    print(f"    判类：{'正类' if p.item() > 0.5 else '负类'}（阈值 0.5）\n")

    # 简短 BCE 训练演示：二分类合成数据
    torch.manual_seed(0)
    x_train = torch.linspace(-1, 5, 40, device=dev).unsqueeze(1)
    y_train = (x_train.squeeze() > 2).float()

    model = nn.Linear(1, 1).to(dev)
    opt = optim.SGD(model.parameters(), lr=0.1)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(150):
        opt.zero_grad()
        loss = bce(model(x_train).squeeze(-1), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        p3 = torch.sigmoid(model(torch.tensor([[3.0]], device=dev))).item()
    print(f"    BCE 训练后 x=3 预测概率 ≈ {p3:.3f}（应接近 0.88 量级）")


def main() -> None:
    seed_all(42)
    dev = device()
    print("线性回归 + 逻辑回归（PyTorch）\n")
    demo_linear_regression(dev)
    demo_logistic_regression(dev)


if __name__ == "__main__":
    main()
