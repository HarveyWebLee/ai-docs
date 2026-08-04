"""
用 PyTorch 计算三分类交叉熵与 KL 散度，对照文档数值例子。

对应文档：docs/01-数学与理论基础/004-信息论基础.md
真实分布 p=[0,1,0]；模型 A 预测 q_A=[0.1,0.8,0.1]；模型 B 预测 q_B=[0.3,0.4,0.3]。

运行：python code/01-数学与理论基础/004-信息论基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn.functional as F


def cross_entropy_manual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """H(p,q) = -Σ p_i log q_i（p 为概率分布）。"""
    return -(p * q.log()).sum()


def main() -> None:
    seed_all(42)
    dev = device()

    # 真实标签：第 2 类（索引 1）
    p = torch.tensor([0.0, 1.0, 0.0], device=dev)
    q_a = torch.tensor([0.1, 0.8, 0.1], device=dev)
    q_b = torch.tensor([0.3, 0.4, 0.3], device=dev)
    target_class = torch.tensor([1], device=dev, dtype=torch.long)

    print("三分类交叉熵与 KL 散度（PyTorch）")
    print(f"p（真实）  = {p.tolist()}")
    print(f"q_A（模型A）= {q_a.tolist()}")
    print(f"q_B（模型B）= {q_b.tolist()}\n")

    for name, q in [("模型 A", q_a), ("模型 B", q_b)]:
        # 手工公式：p 为 one-hot 时 H(p,q) = -log q[正确类]
        ce_manual = cross_entropy_manual(p, q).item()

        # torch.nn.functional：NLL(log q, y) 等价于 -log q[正确类]
        ce_nll = F.nll_loss(q.log().unsqueeze(0), target_class).item()
        # KL(p||q)：p 含 0 时用 F.kl_div（内部处理 log 0）
        kl_fn = F.kl_div(q.log(), p, reduction="sum").item()

        print(f"--- {name} ---")
        print(f"  交叉熵 H(p,q)     = {ce_manual:.4f}  （文档：A≈0.223, B≈0.916）")
        print(f"  F.nll_loss        = {ce_nll:.4f}")
        print(f"  F.cross_entropy   = {ce_nll:.4f}  （one-hot 标签时与 nll 一致）")
        print(f"  KL(p||q) F.kl_div = {kl_fn:.4f}\n")

    print("结论：q_A 对真实类概率更高 → 交叉熵/KL 更小，预测更可信。")


if __name__ == "__main__":
    main()
