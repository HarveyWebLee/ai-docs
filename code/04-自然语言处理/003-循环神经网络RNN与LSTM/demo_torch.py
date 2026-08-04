"""
字符级序列复制任务：用 nn.LSTM 学习「输出 = 输入」，观察 loss 下降。

对应文档：docs/04-自然语言处理/003-循环神经网络RNN与LSTM.md

运行：python code/04-自然语言处理/003-循环神经网络RNN与LSTM/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn
import torch.nn.functional as F


CHARS = list("abcd")
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
SEQ_LEN = 4
HIDDEN = 16
EPOCHS = 80


class CopyLSTM(nn.Module):
    """单步 LSTM：每时刻预测下一字符（复制任务标签=输入本身）。"""

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.fc(out)


def random_batch(batch_size: int, dev: torch.device) -> torch.Tensor:
    """随机字符序列 [B, T]。"""
    return torch.randint(0, len(CHARS), (batch_size, SEQ_LEN), device=dev)


def main() -> None:
    seed_all(42)
    dev = device()

    model = CopyLSTM(len(CHARS), HIDDEN).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    losses: list[float] = []
    for epoch in range(EPOCHS):
        x = random_batch(32, dev)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(CHARS)), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print("任务：LSTM 序列复制（标签 = 输入字符）")
    print(f"字符表：{CHARS}，序列长度：{SEQ_LEN}，训练 {EPOCHS} 轮")
    print(f"初始 loss：{losses[0]:.4f}，最终 loss：{losses[-1]:.4f}")
    print(f"验证：loss 单调下降（末轮 < 首轮）→ {losses[-1] < losses[0]}")
    print(f"验证：最终 loss < 0.5 → {losses[-1] < 0.5}")


if __name__ == "__main__":
    main()
