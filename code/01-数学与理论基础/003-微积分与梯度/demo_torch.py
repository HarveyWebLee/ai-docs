"""
用 PyTorch 自动求导 + 手工 SGD 最小化 f(x)=x²，复现文档迭代表。

对应文档：docs/01-数学与理论基础/003-微积分与梯度.md
初值 x₀=4，学习率 η=0.1，展示 x 逐步趋近 0、f(x) 单调下降。

运行：python code/01-数学与理论基础/003-微积分与梯度/demo_torch.py
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

    x = torch.tensor(4.0, device=dev, requires_grad=True)
    lr = 0.1
    steps = 11  # 打印步 0..10，与文档表格一致

    print("最小化 f(x)=x²（PyTorch autograd + 手工 SGD）")
    print(f"初值 x₀=4，学习率 η={lr}\n")
    print(f"{'步':>3} | {'x':>8} | {'f(x)=x²':>10}")
    print("-" * 28)

    for step in range(steps):
        # 前向：计算损失 f(x)=x²
        loss = x ** 2
        print(f" {step:2d} | {x.item():8.3f} | {loss.item():10.3f}")

        if step == steps - 1:
            break

        # 反向：autograd 求 df/dx = 2x
        loss.backward()

        # 手工 SGD：x ← x - η·∇f（不用 optimizer，便于对照文档手算）
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()

    print(f"\n验证：步 10 时 x≈{x.item():.3f}（文档约 0.429），f(x) 单调下降，x → 0。")


if __name__ == "__main__":
    main()
