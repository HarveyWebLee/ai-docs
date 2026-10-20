"""
缩放点积注意力（Scaled Dot-Product Attention）手写实现并验证 softmax 权重。

对应文档：docs/05-大语言模型与Transformer/001-注意力机制与自注意力.md

运行：python code/05-大语言模型与Transformer/001-注意力机制与自注意力/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    单头缩放点积注意力。
    query/key/value: [batch, seq_len, d_k]
    返回 (输出, 注意力权重)。
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights


def main() -> None:
    seed_all(42)
    dev = device()

    # 单 query 对 3 个 key：q=[1]，keys=[2,1,0] → logits=[2,1,0]
    q = torch.tensor([[[1.0]]], device=dev)
    k = torch.tensor([[[2.0], [1.0], [0.0]]], device=dev)
    v = torch.tensor([[[10.0], [20.0], [30.0]]], device=dev)

    out, weights = scaled_dot_product_attention(q, k, v)
    w = weights.squeeze(0).squeeze(0).tolist()
    expected = [0.665, 0.245, 0.090]

    print("缩放点积注意力 demo")
    print(f"logits（未 softmax）：[2, 1, 0]")
    print(f"softmax 权重：{[round(x, 3) for x in w]}")
    print(f"期望约：{expected}")
    ok = all(abs(w[i] - expected[i]) < 0.01 for i in range(3))
    print(f"验证：三项误差均 < 0.01 → {ok}")
    print(f"加权输出值：{out.squeeze().item():.3f}（≈ 10×0.665 + 20×0.245 + 30×0.090）")


if __name__ == "__main__":
    main()
