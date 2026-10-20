"""
TensorDataset + DataLoader：最小数据管线（PyTorch）。

对应文档：docs/08-AI工程与MLOps/001-数据管线与特征工程.md

运行：python code/08-AI工程与MLOps/001-数据管线与特征工程/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    seed_all(42)
    dev = device()

    n, d = 100, 3
    x = torch.randn(n, d)
    y = (x.sum(dim=1) > 0).long()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("数据管线：TensorDataset → DataLoader 小批量迭代")
    for i, (xb, yb) in enumerate(loader):
        if i >= 2:
            break
        print(f"  batch {i}: X {tuple(xb.shape)}, y {tuple(yb.shape)}")

    total_batches = len(loader)
    print(f"\n共 {n} 样本，batch_size=16 → {total_batches} 个 batch/epoch")


if __name__ == "__main__":
    main()
