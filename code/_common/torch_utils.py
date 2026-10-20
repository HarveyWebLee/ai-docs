"""PyTorch 学习示例的公共小工具（固定随机种子、默认 CPU 设备）。"""

from __future__ import annotations

import random

import torch


def seed_all(seed: int = 42) -> None:
    """固定 Python / PyTorch 随机性，便于教学 demo 复现。"""
    random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    """所有 demo 默认 CPU，保证 CI 无 GPU 也能跑。"""
    return torch.device("cpu")
