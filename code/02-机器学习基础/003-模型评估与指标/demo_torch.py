"""
二分类 MLP + 混淆矩阵 / 精确率 / 召回率 / F1（PyTorch）。

对应文档：docs/02-机器学习基础/003-模型评估与指标.md

运行：python code/02-机器学习基础/003-模型评估与指标/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.optim as optim


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def confusion_and_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    tn = ((y_pred == 0) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else torch.tensor(0.0)
    return {
        "tp": tp.item(),
        "fp": fp.item(),
        "tn": tn.item(),
        "fn": fn.item(),
        "accuracy": acc.item(),
        "precision": prec.item(),
        "recall": rec.item(),
        "f1": f1.item(),
    }


def main() -> None:
    seed_all(42)
    dev = device()

    # 不平衡：90% 负类
    x = torch.linspace(0, 1, 200, device=dev).unsqueeze(1)
    y = (x.squeeze() > 0.55).float()
    n_pos = int(y.sum().item())
    print(f"合成数据：200 样本，正类 {n_pos} / 负类 {200 - n_pos}（不平衡）\n")

    model = TinyMLP().to(dev)
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(300):
        opt.zero_grad()
        logits = model(x).squeeze()
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze())
        preds = (probs >= 0.5).long()
        m = confusion_and_metrics(y.long(), preds)

    print("训练后混淆矩阵与指标（PyTorch MLP）")
    print(f"  TP={m['tp']:.0f} FP={m['fp']:.0f} TN={m['tn']:.0f} FN={m['fn']:.0f}")
    print(f"  准确率={m['accuracy']:.1%}  精确率={m['precision']:.1%}  "
          f"召回率={m['recall']:.1%}  F1={m['f1']:.3f}")
    print("\n提示：类别不平衡时勿只看准确率，需同时看召回率/F1（见文档案例）。")


if __name__ == "__main__":
    main()
