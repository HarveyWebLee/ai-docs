"""
推理模式：model.eval() + torch.no_grad() 批量前向（PyTorch）。

对应文档：docs/08-AI工程与MLOps/003-模型部署与推理服务.md

运行：python code/08-AI工程与MLOps/003-模型部署与推理服务/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def predict_batch(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = model(x)
    return logits.argmax(dim=1)


def main() -> None:
    seed_all(42)
    dev = device()

    model = TinyClassifier().to(dev)
    # 训练态随机 dropout 等在此 demo 无，但 eval() 是部署惯例
    model.train()
    print(f"训练模式 model.training = {model.training}")

    batch = torch.randn(8, 4, device=dev)
    model.eval()
    preds = predict_batch(model, batch)

    print(f"推理模式 model.training = {model.training}")
    print(f"批量输入 shape {tuple(batch.shape)} → 预测类别 {preds.tolist()}")
    print("部署要点：关闭梯度(no_grad)、eval() 固定 BN/Dropout 行为、批量提高吞吐。")


if __name__ == "__main__":
    main()
