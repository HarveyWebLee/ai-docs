"""
单神经元梯度下降一步，复现文档手算数值（w:0→0.8, b:0→0.4, loss 8→2）。

对应文档：docs/03-深度学习基础/002-反向传播与梯度下降.md
模型 ŷ=wx+b，样本 x=2,y=4，lr=0.1，用 autograd + 一步 SGD。

运行：python code/03-深度学习基础/002-反向传播与梯度下降/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def main() -> None:
    seed_all(42)
    dev = device()

    x = torch.tensor(2.0, device=dev)
    y = torch.tensor(4.0, device=dev)
    w = torch.tensor(0.0, device=dev, requires_grad=True)
    b = torch.tensor(0.0, device=dev, requires_grad=True)
    lr = 0.1

    print("单神经元一步 SGD（PyTorch autograd）")
    print("x=2, y=4, 初始 w=0, b=0, lr=0.1\n")
    print("step |   w    |   b    | y_hat  |  loss")
    print("-----+--------+--------+--------+--------")

    for step in range(2):
        y_hat = w * x + b
        loss = 0.5 * (y_hat - y) ** 2
        print(
            f" {step:3d} | {w.item():6.3f} | {b.item():6.3f} | "
            f"{y_hat.item():6.3f} | {loss.item():6.3f}"
        )
        if step == 1:
            break
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

    print("\n验证（文档）：步 0 loss=8；一步后 w=0.8, b=0.4, loss=2。")


if __name__ == "__main__":
    main()
