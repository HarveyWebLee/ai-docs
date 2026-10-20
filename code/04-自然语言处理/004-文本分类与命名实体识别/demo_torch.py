"""
词袋（BoW）+ 线性层：在极小标注句子上做二分类（情感）。

对应文档：docs/04-自然语言处理/004-文本分类与命名实体识别.md

运行：python code/04-自然语言处理/004-文本分类与命名实体识别/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.nn.functional as F


# (句子, 标签)  0=负面  1=正面
SAMPLES: list[tuple[str, int]] = [
    ("这个电影很好看", 1),
    ("剧情精彩推荐", 1),
    ("太无聊了", 0),
    ("浪费时间很差", 0),
    ("演员表演出色", 1),
    ("完全看不下去", 0),
]

EPOCHS = 200


def build_vocab(sentences: list[str]) -> dict[str, int]:
    words = sorted({w for s in sentences for w in s})
    return {w: i for i, w in enumerate(words)}


def sentence_to_bow(sentence: str, vocab: dict[str, int], dim: int) -> torch.Tensor:
    """字符级词袋（中文短句按字切分）。"""
    vec = torch.zeros(dim)
    for ch in sentence:
        if ch in vocab:
            vec[vocab[ch]] += 1.0
    return vec


class BoWClassifier(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(vocab_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main() -> None:
    seed_all(42)
    dev = device()

    sentences = [s for s, _ in SAMPLES]
    labels = torch.tensor([y for _, y in SAMPLES], device=dev)
    vocab = build_vocab(sentences)
    dim = len(vocab)

    X = torch.stack([sentence_to_bow(s, vocab, dim) for s in sentences]).to(dev)
    model = BoWClassifier(dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(EPOCHS):
        logits = model(X)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    test_sent = "这部电影非常精彩"
    test_bow = sentence_to_bow(test_sent, vocab, dim).unsqueeze(0).to(dev)
    test_pred = model(test_bow).argmax(dim=1).item()
    label_names = ["负面", "正面"]

    print("训练样本：")
    for s, y in SAMPLES:
        print(f"  [{label_names[y]}] {s}")
    print(f"词表大小：{dim}，训练 {EPOCHS} 轮，训练集准确率：{acc:.2%}")
    print(f"测试句：「{test_sent}」→ 预测：{label_names[test_pred]}")
    print(f"验证：训练集全对 → {acc == 1.0}")


if __name__ == "__main__":
    main()
