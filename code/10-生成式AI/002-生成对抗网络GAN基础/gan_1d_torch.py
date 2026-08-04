"""
一维 GAN：生成器+判别器 MLP，少量步训练后 G 输出均值向真实分布 0 靠拢。

对应文档：docs/10-生成式AI/002-生成对抗网络GAN基础.md

运行：python code/10-生成式AI/002-生成对抗网络GAN基础/gan_1d_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    seed_all(42)
    dev = device()

    g = Generator().to(dev)
    d = Discriminator().to(dev)
    # 略大的 G 学习率，让生成均值更快靠近真实 0
    opt_g = torch.optim.Adam(g.parameters(), lr=0.005)
    opt_d = torch.optim.Adam(d.parameters(), lr=0.002)
    bce = nn.BCELoss()

    with torch.no_grad():
        init_mean = g(torch.randn(256, 1, device=dev)).mean().item()

    print("一维 GAN（真实数据 N(0,0.3²)，G 从噪声映射到 1D）")
    print(f"初始 G 输出均值: {init_mean:.3f}\n")
    print(f"{'step':>4} | {'G均值':>8} | {'D(真)':>6} | {'D(假)':>6}")
    print("-" * 34)

    for step in range(61):
        real = torch.randn(128, 1, device=dev) * 0.3
        z = torch.randn(128, 1, device=dev)

        # 训练 D
        fake_detached = g(z).detach()
        d.zero_grad()
        loss_d = bce(d(real), torch.ones(128, 1, device=dev)) + bce(
            d(fake_detached), torch.zeros(128, 1, device=dev)
        )
        loss_d.backward()
        opt_d.step()

        # 训练 G（2 步 / D 1 步，稳定对抗）
        for _ in range(2):
            z_g = torch.randn(128, 1, device=dev)
            fake = g(z_g)
            g.zero_grad()
            loss_g = bce(d(fake), torch.ones(128, 1, device=dev))
            loss_g.backward()
            opt_g.step()

        if step % 10 == 0:
            with torch.no_grad():
                g_mean = g(torch.randn(512, 1, device=dev)).mean().item()
                d_real = d(real).mean().item()
                d_fake = d(fake_detached).mean().item()
            print(f"{step:4d} | {g_mean:8.3f} | {d_real:6.3f} | {d_fake:6.3f}")

    with torch.no_grad():
        final_mean = g(torch.randn(512, 1, device=dev)).mean().item()

    moved_toward_zero = abs(final_mean) < abs(init_mean)
    print(f"\n最终 G 均值: {final_mean:.3f}（目标≈0）")
    print(f"向 0 靠拢: {moved_toward_zero}（|{final_mean:.3f}| < |{init_mean:.3f}|）")


if __name__ == "__main__":
    main()
