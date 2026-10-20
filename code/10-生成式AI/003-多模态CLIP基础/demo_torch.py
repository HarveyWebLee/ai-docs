"""
CLIP 式 L2 归一化 + 矩阵乘法求图文余弦相似度（PyTorch）。

对应文档：docs/10-生成式AI/003-多模态CLIP基础.md

运行：python code/10-生成式AI/003-多模态CLIP基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn.functional as F


def main() -> None:
    seed_all(42)
    dev = device()

    # 3 图像 × 3 文本，3 维嵌入（与标准库 demo 语义对齐）
    image_emb = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.85, 0.1],
            [0.0, 0.1, 0.95],
        ],
        device=dev,
    )
    text_emb = torch.tensor(
        [
            [0.88, 0.15, 0.05],
            [0.12, 0.9, 0.08],
            [0.05, 0.08, 0.92],
        ],
        device=dev,
    )

    img_n = F.normalize(image_emb, dim=1)
    txt_n = F.normalize(text_emb, dim=1)
    sim = img_n @ txt_n.T  # 余弦相似度矩阵

    labels = ["cat", "dog", "car"]
    print("CLIP 相似度矩阵（PyTorch normalize + matmul）\n")
    print("       " + "  ".join(f"{t:>8}" for t in labels))
    for i, name in enumerate(["I_cat", "I_dog", "I_car"]):
        row = "  ".join(f"{sim[i, j].item():>8.3f}" for j in range(3))
        print(f"{name:>6} {row}")

    print("\n零样本分类（每行 argmax）：")
    for i, name in enumerate(["I_cat", "I_dog", "I_car"]):
        j = int(sim[i].argmax().item())
        print(f"  {name} → {labels[j]}（sim={sim[i, j].item():.3f}）")


if __name__ == "__main__":
    main()
