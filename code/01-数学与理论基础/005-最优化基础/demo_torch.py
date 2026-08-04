"""
比较 SGD / Momentum / Adam 在狭长峡谷损失面上的 200 步收敛。

对应文档：docs/01-数学与理论基础/005-最优化基础.md
损失 J(w1,w2) = 0.1·w1² + w2²，起点 (5,5)，lr=0.1。

运行：python code/01-数学与理论基础/005-最优化基础/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.optim as optim


def make_params() -> torch.Tensor:
    """可训练参数向量 [w1, w2]。"""
    return torch.tensor([5.0, 5.0], requires_grad=True)


def loss_fn(params: torch.Tensor) -> torch.Tensor:
    w1, w2 = params[0], params[1]
    return 0.1 * w1 ** 2 + w2 ** 2


def run_optimizer(name: str, opt_cls, steps: int = 200, **kwargs) -> tuple[str, float]:
    seed_all(42)
    params = make_params()
    optimizer = opt_cls([params], **kwargs)

    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return name, final_loss


def main() -> None:
    dev = device()
    steps = 200
    lr = 0.1

    print("二维峡谷损失 J = 0.1·w1² + w2²（PyTorch 优化器）")
    print(f"起点 (5, 5)，lr={lr}，共 {steps} 步\n")
    print(f"{'优化器':<18} | {'最终损失':>12}")
    print("-" * 34)

    results = [
        run_optimizer("SGD", optim.SGD, steps, lr=lr),
        run_optimizer("SGD + Momentum", optim.SGD, steps, lr=lr, momentum=0.9),
        run_optimizer("Adam", optim.Adam, steps, lr=lr),
    ]

    for name, final_loss in results:
        print(f"{name:<18} | {final_loss:>12.6f}")

    print("\n结论：w2 方向更陡，Momentum / Adam 最终损失通常低于纯 SGD。")


if __name__ == "__main__":
    main()
