"""
固定随机嵌入矩阵 + 余弦检索：为问题返回 top-1 最相关文档句。

对应文档：docs/05-大语言模型与Transformer/004-检索增强生成RAG.md

运行：python code/05-大语言模型与Transformer/004-检索增强生成RAG/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn.functional as F


DOCS = [
    "PyTorch 是流行的深度学习框架",
    "Transformer 使用自注意力机制",
    "RAG 先检索再生成答案",
    "词向量把词映射到连续向量空间",
]

QUESTION = "什么是 RAG 检索增强生成"


def embed_texts(texts: list[str], table: torch.Tensor, char2idx: dict[str, int]) -> torch.Tensor:
    """字符平均池化查固定嵌入表（教学用简化编码器）。"""
    dim = table.size(1)
    vecs = []
    for text in texts:
        ids = [char2idx[c] for c in text if c in char2idx]
        if not ids:
            vecs.append(torch.zeros(dim, device=table.device))
        else:
            vecs.append(table[torch.tensor(ids, device=table.device)].mean(dim=0))
    return torch.stack(vecs)


def retrieve_top1(query: torch.Tensor, doc_embs: torch.Tensor) -> int:
    sims = F.cosine_similarity(query.unsqueeze(0), doc_embs)
    return int(sims.argmax().item())


def main() -> None:
    seed_all(42)
    dev = device()

    chars = sorted({c for t in DOCS + [QUESTION] for c in t})
    char2idx = {c: i for i, c in enumerate(chars)}

    embed_dim = 16
    table = torch.randn(len(chars), embed_dim, device=dev)

    doc_embs = embed_texts(DOCS, table, char2idx)
    q_emb = embed_texts([QUESTION], table, char2idx).squeeze(0)

    top_idx = retrieve_top1(q_emb, doc_embs)
    sims = F.cosine_similarity(q_emb.unsqueeze(0), doc_embs).tolist()

    print("RAG 检索 demo（固定嵌入表 + 余弦相似度）")
    print(f"问题：{QUESTION}")
    print("文档库：")
    for i, doc in enumerate(DOCS):
        print(f"  [{i}] {doc}  (sim={sims[i]:.4f})")
    print(f"Top-1 检索结果：[{top_idx}] {DOCS[top_idx]}")
    print(f"验证：命中 RAG 相关句 → {top_idx == 2}")


if __name__ == "__main__":
    main()
