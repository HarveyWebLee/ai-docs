"""
BatchNorm 对 batch 均值/方差的影响，以及 Dropout train/eval 模式差异。

对应文档：docs/03-深度学习基础/004-正则化与批归一化.md

运行：python code/03-深度学习基础/004-正则化与批归一化/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


def demo_batch_norm(dev: torch.device) -> None:
    torch.manual_seed(0)
    # 小 batch：均值/方差偏离 0/1
    x = torch.randn(8, 4, device=dev) * 2.0 + 3.0
    bn = nn.BatchNorm1d(4).to(dev)

    print("(a) BatchNorm 效果")
    print(f"    输入 batch 形状 {tuple(x.shape)}")
    print(f"    输入 mean≈{x.mean(dim=0).mean().item():.3f}, var≈{x.var(dim=0, unbiased=False).mean().item():.3f}")

    bn.train()
    y = bn(x)
    print(f"    BN 后 mean≈{y.mean(dim=0).abs().mean().item():.4f}（应接近 0）")
    print(f"    BN 后 var≈{y.var(dim=0, unbiased=False).mean().item():.4f}（应接近 1）\n")


def demo_dropout(dev: torch.device) -> None:
    torch.manual_seed(1)
    x = torch.ones(4, 8, device=dev)
    drop = nn.Dropout(p=0.5)

    drop.train()
    y_train = drop(x)
    drop.eval()
    y_eval = drop(x)

    zero_ratio = (y_train == 0).float().mean().item()
    print("(b) Dropout train vs eval")
    print(f"    p=0.5，输入全 1，形状 {tuple(x.shape)}")
    print(f"    train 置零比例 ≈ {zero_ratio:.2f}（约一半神经元被丢弃）")
    print(f"    train 输出均值 = {y_train.mean().item():.4f}（幸存值×1/(1-p)，期望仍≈1）")
    print(f"    eval  输出均值 = {y_eval.mean().item():.4f}（关闭 dropout，原样通过）\n")


def main() -> None:
    seed_all(42)
    dev = device()
    print("BatchNorm + Dropout 演示（PyTorch）\n")
    demo_batch_norm(dev)
    demo_dropout(dev)
    print("结论：BN 归一化 batch 统计量；Dropout 仅在 train 时随机丢弃。")


if __name__ == "__main__":
    main()
