"""
一维 DDPM 风格前向加噪：x₀=1.0，打印各步 x_t 的均值/方差趋势。

对应文档：docs/10-生成式AI/001-扩散模型Diffusion基础.md

运行：python code/10-生成式AI/001-扩散模型Diffusion基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def forward_diffusion_1d(
    x0: torch.Tensor,
    betas: torch.Tensor,
    t: int,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)"""
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    ab = alpha_bar[t - 1]
    if noise is None:
        noise = torch.randn_like(x0)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def main() -> None:
    seed_all(42)
    dev = device()

    x0 = torch.tensor([1.0], device=dev)
    betas = torch.tensor([0.1, 0.2, 0.3], device=dev)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    print("一维 DDPM 前向加噪（x₀=1.0）")
    print(f"{'t':>3} | {'ᾱ_t':>6} | {'E[x_t]':>8} | {'Var[x_t]':>10} | x_t(采样)")
    print("-" * 52)
    print(f"  0 | {'1.000':>6} | {'1.0000':>8} | {'0.000000':>10} | {x0.item():.4f}")

    for t in range(1, len(betas) + 1):
        ab = alpha_bar[t - 1]
        mean = torch.sqrt(ab) * x0
        var = 1.0 - ab
        noise = torch.randn(1, device=dev)
        xt = forward_diffusion_1d(x0, betas, t, noise=noise)
        print(
            f"  {t} | {ab.item():>6.3f} | {mean.item():>8.4f} | {var.item():>10.6f} | {xt.item():>+.4f}"
        )

    print("\n趋势：t 增大 → ᾱ_t 减小 → 均值趋 0、方差趋 1（接近纯噪声）")


if __name__ == "__main__":
    main()
