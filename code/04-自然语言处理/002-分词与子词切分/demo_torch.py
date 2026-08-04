"""
字符级词表 + nn.Embedding：演示 token id 与向量查表（BPE 前置概念）。

对应文档：docs/04-自然语言处理/002-分词与子词切分.md

运行：python code/04-自然语言处理/002-分词与子词切分/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


def char_tokenize(text: str) -> list[str]:
    return list(text)


def main() -> None:
    seed_all(42)
    dev = device()

    corpus = ["low", "lower", "newest", "widest"]
    chars = sorted({c for w in corpus for c in w})
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size, embed_dim = len(stoi), 8

    embedding = nn.Embedding(vocab_size, embed_dim).to(dev)

    word = "newest"
    tokens = char_tokenize(word)
    ids = torch.tensor([stoi[c] for c in tokens], device=dev)
    vectors = embedding(ids)

    print("字符级 token → id → Embedding 向量（子词/BPE 前的最小闭环）")
    print(f"词表大小 |V|={vocab_size}，嵌入维度 d={embed_dim}")
    print(f"词 '{word}' → tokens {tokens} → ids {ids.tolist()}")
    print(f"Embedding 输出 shape: {tuple(vectors.shape)}  # (seq_len, d)")
    print(f"首 token '{tokens[0]}' 向量前 4 维: {vectors[0, :4].tolist()}")


if __name__ == "__main__":
    main()
