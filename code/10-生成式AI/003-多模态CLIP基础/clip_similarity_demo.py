"""
CLIP 式图文相似度与零样本分类演示（手工嵌入向量）。

对应文档：docs/10-生成式AI/003-多模态CLIP基础.md
用 3 维向量模拟图像/文本嵌入，L2 归一化后点积 = 余弦相似度。

运行：python code/10-生成式AI/003-多模态CLIP基础/clip_similarity_demo.py
（仅依赖标准库）
"""

import math

# 手工「嵌入」：3 个图像、3 个文本（维度已大致对齐语义）
IMAGE_EMB = {
    "I1_cat": [0.9, 0.1, 0.0],
    "I2_dog": [0.1, 0.85, 0.1],
    "I3_car": [0.0, 0.1, 0.95],
}

TEXT_EMB = {
    "T1_a cat": [0.88, 0.15, 0.05],
    "T2_a dog": [0.12, 0.9, 0.08],
    "T3_a car": [0.05, 0.08, 0.92],
}

LABELS = ["cat", "dog", "car"]


def l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-9:
        return v[:]
    return [x / n for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def similarity(a: list[float], b: list[float]) -> float:
    return dot(l2_normalize(a), l2_normalize(b))


def main() -> None:
    print("CLIP 式相似度矩阵（手工 3 维嵌入）\n")
    text_keys = list(TEXT_EMB.keys())
    img_keys = list(IMAGE_EMB.keys())

    header = "          " + "".join(f"{tk[:8]:>10}" for tk in text_keys)
    print(header)
    for ik in img_keys:
        row = f"{ik[:8]:>10}"
        for tk in text_keys:
            s = similarity(IMAGE_EMB[ik], TEXT_EMB[tk])
            row += f"{s:>10.3f}"
        print(row)

    print("\n零样本分类（图像 vs 类别 prompt）：")
    prompts = {label: TEXT_EMB[f"T{i+1}_a {label}"] for i, label in enumerate(LABELS)}
    for ik in img_keys:
        scores = {label: similarity(IMAGE_EMB[ik], prompts[label]) for label in LABELS}
        best = max(scores, key=scores.get)
        print(f"  {ik} -> 预测「{best}」（分数 {scores[best]:.3f}）")


if __name__ == "__main__":
    main()
