"""
偏好优化损失：-log σ(r_chosen - r_rejected)，与 RLHF 中 reward 建模思路一致。

对应文档：docs/05-大语言模型与Transformer/005-对齐与RLHF.md

运行：python code/05-大语言模型与Transformer/005-对齐与RLHF/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn.functional as F


def preference_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry / DPO 风格的 pairwise 偏好损失（单样本）。"""
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def main() -> None:
    seed_all(42)
    dev = device()

    # 玩具 reward 分数：chosen 应高于 rejected
    pairs = [
        (2.5, 0.5),
        (1.0, -0.5),
        (3.0, 2.8),  # 边际较小，loss 更大
    ]

    print("RLHF 偏好损失 demo：L = -log σ(r_chosen - r_rejected)")
    total = 0.0
    for i, (rc, rr) in enumerate(pairs):
        r_c = torch.tensor([rc], device=dev)
        r_r = torch.tensor([rr], device=dev)
        loss = preference_loss(r_c, r_r).item()
        total += loss
        margin = rc - rr
        print(f"  样本 {i + 1}: r_chosen={rc}, r_rejected={rr}, 边际={margin:.1f}, loss={loss:.4f}")

    avg_loss = total / len(pairs)
    # 边际为负时应得到较大 loss
    bad_c = torch.tensor([-1.0], device=dev)
    bad_r = torch.tensor([2.0], device=dev)
    bad_loss = preference_loss(bad_c, bad_r).item()

    print(f"平均 loss（合理偏好对）：{avg_loss:.4f}")
    print(f"错误偏好 (chosen<rejected) loss：{bad_loss:.4f}")
    print(f"验证：错误偏好 loss > 平均 loss → {bad_loss > avg_loss}")


if __name__ == "__main__":
    main()
