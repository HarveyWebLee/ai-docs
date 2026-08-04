"""
ViT Patch Embedding：8×8 图像切 4 个 4×4 patch，线性投影到 D 维。

对应文档：docs/06-计算机视觉/005-视觉Transformer ViT.md

运行：python code/06-计算机视觉/005-视觉Transformer ViT/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B,C,H,W) → (B, N, P²·C)。"""
    b, c, h, w = x.shape
    p = patch_size
    x = x.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b, (h // p) * (w // p), c * p * p)


def main() -> None:
    seed_all(42)
    dev = device()

    patch_size = 4
    embed_dim = 8
    # 8×8 灰度图，四个象限不同亮度便于区分 patch
    image = torch.arange(64, dtype=torch.float32, device=dev).view(1, 1, 8, 8) / 64.0

    patches = patchify(image, patch_size)
    proj = nn.Linear(patch_size * patch_size * 1, embed_dim, bias=True).to(dev)

    with torch.no_grad():
        embeddings = proj(patches)

    n_patches = (8 // patch_size) ** 2
    print("ViT Patch Embedding 演示")
    print(f"输入图像 shape: {tuple(image.shape)}  (8×8 灰度)")
    print(f"patch 大小: {patch_size}×{patch_size} → 共 {n_patches} 个 patch")
    print(f"展平后 patches shape: {tuple(patches.shape)}  (每 patch {patch_size**2} 维)")
    print(f"线性投影 → embed_dim={embed_dim}")
    print(f"embeddings shape: {tuple(embeddings.shape)}")
    print(f"\n第 1 个 patch 展平前 8 维: {patches[0, 0, :8].tolist()}")
    print(f"第 1 个 patch embedding:   {embeddings[0, 0].tolist()}")


if __name__ == "__main__":
    main()
