"""
LoRA 线性层：冻结 W₀，只训练低秩 BA，演示一步前向+反向。

对应文档：docs/10-生成式AI/005-LoRA低秩微调.md

运行：python code/10-生成式AI/005-LoRA低秩微调/lora_layer_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """h = W₀x + (α/r) BAx，W₀ 冻结，只训 A、B。"""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + (self.alpha / self.rank) * lora_out


def main() -> None:
    seed_all(42)
    dev = device()

    layer = LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0).to(dev)
    x = torch.tensor([[1.0, 0.5, -0.2, 0.3]], device=dev)
    target = torch.tensor([[1.0, 0.0]], device=dev)

    w0_before = layer.base.weight.clone()
    a_before = layer.lora_a.clone()
    b_before = layer.lora_b.clone()

    out = layer(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    w0_after = layer.base.weight
    a_grad_norm = layer.lora_a.grad.norm().item()
    w0_grad = layer.base.weight.grad

    print("LoRA 线性层：h = W₀x + (α/r)BAx")
    print(f"输入 x: {x.squeeze().tolist()}")
    print(f"前向输出: {out.squeeze().tolist()}")
    print(f"loss MSE: {loss.item():.6f}")
    print(f"\nW₀ 是否冻结（更新前后相同）: {torch.equal(w0_before, w0_after)}")
    print(f"W₀.grad 为 None: {w0_grad is None}")
    print(f"LoRA A 梯度范数: {a_grad_norm:.6f}（>0 说明 BA 在训练）")
    print(f"LoRA A 有 grad: {layer.lora_a.grad is not None and layer.lora_a.grad.abs().sum() > 0}")

    opt = torch.optim.SGD(
        [layer.lora_a, layer.lora_b],
        lr=0.5,
    )
    opt.step()
    print(f"一步 SGD 后 B 相对初值变化: {(layer.lora_b - b_before).norm().item():.6f}")


if __name__ == "__main__":
    main()
