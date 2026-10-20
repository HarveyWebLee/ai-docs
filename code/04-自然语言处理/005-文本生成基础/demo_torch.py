"""
字符级 GRU 下一字预测（PyTorch 版 char_lm 概念）。

对应文档：docs/04-自然语言处理/005-文本生成基础.md

运行：python code/04-自然语言处理/005-文本生成基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.nn.functional as F


CORPUS = ["我爱中国", "中国人民", "中国爱我"]
EPOCHS = 120
HIDDEN = 32


def build_char_vocab(texts: list[str]) -> tuple[list[str], dict[str, int]]:
    chars = sorted({c for t in texts for c in t})
    return chars, {c: i for i, c in enumerate(chars)}


class CharGRU(nn.Module):
    """给定前缀字符，预测下一个字符。"""

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.gru(emb)
        return self.fc(out)


def make_training_pairs(
    texts: list[str], char2idx: dict[str, int], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for text in texts:
        ids = [char2idx[c] for c in text]
        for i in range(len(ids) - 1):
            xs.append(ids[i])
            ys.append(ids[i + 1])
    return (
        torch.tensor(xs, device=dev).unsqueeze(1),
        torch.tensor(ys, device=dev),
    )


@torch.no_grad()
def predict_next(model: CharGRU, prefix: str, char2idx: dict[str, int], idx2char: list[str], dev: torch.device) -> str:
    if not prefix:
        return "?"
    x = torch.tensor([[char2idx[prefix[-1]]]], device=dev)
    logits = model(x)
    nxt = logits.argmax(dim=-1).item()
    return idx2char[nxt]


def main() -> None:
    seed_all(42)
    dev = device()

    chars, char2idx = build_char_vocab(CORPUS)
    idx2char = chars
    X, Y = make_training_pairs(CORPUS, char2idx, dev)

    model = CharGRU(len(chars), HIDDEN).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.08)

    losses: list[float] = []
    for _ in range(EPOCHS):
        logits = model(X).squeeze(1)
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print("训练语料：", CORPUS)
    print(f"字符表：{chars}，GRU hidden={HIDDEN}，训练 {EPOCHS} 轮")
    print(f"初始 loss：{losses[0]:.4f}，最终 loss：{losses[-1]:.4f}")

    for prefix in ["我", "中", "中国"]:
        nxt = predict_next(model, prefix, char2idx, idx2char, dev)
        print(f"  前缀「{prefix}」→ 预测下一字「{nxt}」")

    print(f"验证：loss 下降 → {losses[-1] < losses[0]}")


if __name__ == "__main__":
    main()
