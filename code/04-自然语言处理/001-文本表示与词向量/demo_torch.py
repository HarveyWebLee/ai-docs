"""
nn.Embedding 查表 + 余弦相似度：语义相近的词索引距离更近。

对应文档：docs/04-自然语言处理/001-文本表示与词向量.md

运行：python code/04-自然语言处理/001-文本表示与词向量/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.nn.functional as F


# 小词表：动物词彼此相近，交通工具词彼此相近
VOCAB = ["猫", "狗", "老虎", "汽车", "飞机", "火车"]
# 手工构造的 6×4 嵌入（已归一化方向），便于演示相似度
RAW_VECTORS = torch.tensor(
    [
        [1.0, 0.9, 0.0, 0.0],   # 猫
        [0.95, 1.0, 0.05, 0.0],  # 狗
        [0.85, 0.8, 0.1, 0.0],   # 老虎
        [0.0, 0.0, 1.0, 0.9],    # 汽车
        [0.0, 0.05, 0.95, 1.0],  # 飞机
        [0.0, 0.0, 0.9, 0.95],   # 火车
    ],
    dtype=torch.float32,
)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """两向量余弦相似度（标量）。"""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main() -> None:
    seed_all(42)
    dev = device()

    embedding = nn.Embedding(num_embeddings=len(VOCAB), embedding_dim=4)
    embedding.weight.data.copy_(RAW_VECTORS)
    embedding = embedding.to(dev)

    idx_cat = VOCAB.index("猫")
    idx_dog = VOCAB.index("狗")
    idx_car = VOCAB.index("汽车")

    vec_cat = embedding(torch.tensor([idx_cat], device=dev)).squeeze(0)
    vec_dog = embedding(torch.tensor([idx_dog], device=dev)).squeeze(0)
    vec_car = embedding(torch.tensor([idx_car], device=dev)).squeeze(0)

    sim_cat_dog = cosine_sim(vec_cat, vec_dog)
    sim_cat_car = cosine_sim(vec_cat, vec_car)

    print("词表：", VOCAB)
    print(f"Embedding 权重形状：{tuple(embedding.weight.shape)}")
    print(f"「猫」向量（查表 idx={idx_cat}）：{vec_cat.tolist()}")
    print(f"余弦相似度 猫-狗：{sim_cat_dog:.4f}")
    print(f"余弦相似度 猫-汽车：{sim_cat_car:.4f}")
    print(f"验证：动物对相似度 > 跨类相似度 → {sim_cat_dog > sim_cat_car}")


if __name__ == "__main__":
    main()
