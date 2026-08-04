"""
大 MLP 在小数据集上过拟合 vs weight_decay 正则化对比。

对应文档：docs/02-机器学习基础/004-过拟合与正则化.md
训练集仅 12 点，隐层 64 维 MLP；对比无正则与 weight_decay=0.01 的 train/test 差距。

运行：python code/02-机器学习基础/004-过拟合与正则化/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim


def make_data(n_train: int = 12, n_test: int = 40) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """一维回归：y = sin(x) + 噪声。"""
    torch.manual_seed(7)
    x_train = torch.linspace(-1, 1, n_train).unsqueeze(1)
    y_train = torch.sin(x_train * 3.0) + torch.randn_like(x_train) * 0.05

    x_test = torch.linspace(-1, 1, n_test).unsqueeze(1)
    y_test = torch.sin(x_test * 3.0)  # 测试集无噪声
    return x_train, y_train, x_test, y_test


class LargeMLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_eval(weight_decay: float, epochs: int = 800) -> tuple[float, float]:
    seed_all(42)
    dev = device()
    x_tr, y_tr, x_te, y_te = make_data()
    x_tr, y_tr = x_tr.to(dev), y_tr.to(dev)
    x_te, y_te = x_te.to(dev), y_te.to(dev)

    model = LargeMLP().to(dev)
    opt = optim.Adam(model.parameters(), lr=0.05, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_tr), y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_mse = loss_fn(model(x_tr), y_tr).item()
        test_mse = loss_fn(model(x_te), y_te).item()
    return train_mse, test_mse


def main() -> None:
    print("过拟合 vs L2 正则（weight_decay，PyTorch）")
    print("小训练集 12 点 + 大 MLP（隐层 64×2）\n")

    tr0, te0 = train_and_eval(weight_decay=0.0)
    tr1, te1 = train_and_eval(weight_decay=0.01)

    print(f"{'设置':<22} | {'训练 MSE':>10} | {'测试 MSE':>10} | {'Test-Train 差距':>14}")
    print("-" * 64)
    print(f"{'无正则 (wd=0)':<22} | {tr0:>10.4f} | {te0:>10.4f} | {te0 - tr0:>14.4f}")
    print(f"{'weight_decay=0.01':<22} | {tr1:>10.4f} | {te1:>10.4f} | {te1 - tr1:>14.4f}")

    print("\n结论：无正则时训练误差极低但测试误差高（过拟合）；")
    print("      weight_decay 增大测试泛化，train/test 差距通常缩小。")


if __name__ == "__main__":
    main()
