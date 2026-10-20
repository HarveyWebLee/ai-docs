"""
在 J=0.1·w1²+w2² 上比较 SGD / Momentum / Adam 达到 loss<1e-3 的步数。

对应文档：docs/03-深度学习基础/003-优化器与学习率调度.md
与 optimizer_compare.py 同一损失面，用 torch.optim 实现。

运行：python code/03-深度学习基础/003-优化器与学习率调度/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.optim as optim


def loss_fn(params: torch.Tensor) -> torch.Tensor:
    w1, w2 = params[0], params[1]
    return 0.1 * w1 ** 2 + w2 ** 2


def steps_to_threshold(opt_cls, max_steps: int = 500, lr: float = 0.1, thresh: float = 1e-3, **kwargs) -> tuple[int, float]:
    seed_all(42)
    params = torch.tensor([5.0, 5.0], requires_grad=True)
    optimizer = opt_cls([params], lr=lr, **kwargs)

    final_loss = float("inf")
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if final_loss < thresh:
            return step, final_loss
    return max_steps, final_loss


def main() -> None:
    thresh = 1e-3
    max_steps = 500
    lr = 0.1

    print("优化器收敛步数对比（PyTorch）")
    print(f"损失 J = 0.1·w1² + w2²，起点 (5,5)，lr={lr}")
    print(f"收敛判据：loss < {thresh}\n")
    print(f"{'优化器':<18} | {'步数':>8} | {'最终 loss':>12}")
    print("-" * 44)

    configs = [
        ("SGD", optim.SGD, {}),
        ("SGD + Momentum", optim.SGD, {"momentum": 0.9}),
        ("Adam", optim.Adam, {}),
    ]

    for name, cls, kw in configs:
        steps, final = steps_to_threshold(cls, max_steps, lr, thresh, **kw)
        status = f"{steps}" if final < thresh else f">{max_steps}"
        print(f"{name:<18} | {status:>8} | {final:>12.6f}")

    print("\n结论：峡谷面上 Momentum / Adam 通常更少步数达到 loss 阈值。")


if __name__ == "__main__":
    main()
