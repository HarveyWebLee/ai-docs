"""
残差块 y = x + F(x)：F≈0 时输出≈输入，演示 skip connection 恒等路径。

对应文档：docs/06-计算机视觉/003-经典CNN架构与残差网络.md

运行：python code/06-计算机视觉/003-经典CNN架构与残差网络/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """y = x + F(x)，F 为两层 1×1 卷积（演示用）。"""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


def main() -> None:
    seed_all(42)
    dev = device()

    block = ResidualBlock(4).to(dev)
    x = torch.randn(1, 4, 2, 2, device=dev)

    # 未训练：F(x) 随机，输出与输入有差
    with torch.no_grad():
        y_before = block(x)
        diff_before = (y_before - x).abs().mean().item()

    # 把 F 的最后一层权重置零 → F(x)≈0，y≈x（恒等路径）
    with torch.no_grad():
        block.f[-1].weight.zero_()
        block.f[-1].bias.zero_()
        y_after = block(x)
        diff_after = (y_after - x).abs().mean().item()

    print("残差块 y = x + F(x)")
    print(f"输入 x 切片: {x[0, 0].tolist()}")
    print(f"F≈0 前 |y-x| 均值: {diff_before:.6f}")
    print(f"F≈0 后 |y-x| 均值: {diff_after:.6f}")
    print(f"F≈0 后输出≈输入: {torch.allclose(y_after, x, atol=1e-5)}")


if __name__ == "__main__":
    main()
