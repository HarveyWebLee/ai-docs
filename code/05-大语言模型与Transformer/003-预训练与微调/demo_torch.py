"""
LoRA 风格低秩适配：y = W x + (B A) x，可训练参数量远小于全量 W。

对应文档：docs/05-大语言模型与Transformer/003-预训练与微调.md

运行：python code/05-大语言模型与Transformer/003-预训练与微调/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """冻结全秩 W，仅训练低秩 A∈R^{r×in}、B∈R^{out×r}。"""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.T @ self.lora_b.T)


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main() -> None:
    seed_all(42)
    dev = device()

    in_f, out_f, rank = 768, 768, 8
    lora_layer = LoRALinear(in_f, out_f, rank).to(dev)
    full_layer = nn.Linear(in_f, out_f, bias=False).to(dev)

    full_params = sum(p.numel() for p in full_layer.parameters())
    trainable = count_trainable(lora_layer)
    frozen = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)

    x = torch.randn(2, in_f, device=dev)
    y = lora_layer(x)

    print("LoRA 低秩适配 demo（模拟 768×768 线性层）")
    print(f"全量 Linear 参数量：{full_params:,}")
    print(f"LoRA 可训练参数量：{trainable:,}  (r={rank} → r×(in+out)={rank * (in_f + out_f):,})")
    print(f"LoRA 冻结 base 参数量：{frozen:,}")
    print(f"可训练 / 全量比例：{trainable / full_params * 100:.2f}%")
    print(f"前向输出形状：{tuple(y.shape)}")
    print(f"验证：可训练参 << 全量 → {trainable < full_params // 10}")


if __name__ == "__main__":
    main()
