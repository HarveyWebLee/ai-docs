"""
在二维"狭长峡谷"损失面上比较 SGD、Momentum、Adam 的收敛步数。

损失：J(w1, w2) = 0.1 * w1^2 + w2^2
梯度：∇J = (0.2 * w1, 2 * w2)

w2 方向比 w1 陡 10 倍，纯 SGD 易震荡；Momentum / Adam 通常更快到达原点附近。

对应文档：docs/03-深度学习基础/003-优化器与学习率调度.md

运行：python code/03-深度学习基础/003-优化器与学习率调度/optimizer_compare.py
（仅依赖标准库）
"""

import math


def loss_and_grad(w1: float, w2: float) -> tuple[float, float, float]:
    """返回 (损失, dJ/dw1, dJ/dw2)。"""
    j = 0.1 * w1 * w1 + w2 * w2
    g1 = 0.2 * w1
    g2 = 2.0 * w2
    return j, g1, g2


def run_sgd(steps: int, lr: float) -> tuple[int, float]:
    w1, w2 = 5.0, 5.0
    for i in range(steps):
        j, g1, g2 = loss_and_grad(w1, w2)
        if j < 1e-4:
            return i, j
        w1 -= lr * g1
        w2 -= lr * g2
    return steps, loss_and_grad(w1, w2)[0]


def run_momentum(steps: int, lr: float, beta: float = 0.9) -> tuple[int, float]:
    w1, w2 = 5.0, 5.0
    v1, v2 = 0.0, 0.0
    for i in range(steps):
        j, g1, g2 = loss_and_grad(w1, w2)
        if j < 1e-4:
            return i, j
        v1 = beta * v1 + g1
        v2 = beta * v2 + g2
        w1 -= lr * v1
        w2 -= lr * v2
    return steps, loss_and_grad(w1, w2)[0]


def run_adam(
    steps: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[int, float]:
    w1, w2 = 5.0, 5.0
    m1 = m2 = v1 = v2 = 0.0
    for t in range(1, steps + 1):
        j, g1, g2 = loss_and_grad(w1, w2)
        if j < 1e-4:
            return t - 1, j

        m1 = beta1 * m1 + (1 - beta1) * g1
        m2 = beta1 * m2 + (1 - beta1) * g2
        v1 = beta2 * v1 + (1 - beta2) * g1 * g1
        v2 = beta2 * v2 + (1 - beta2) * g2 * g2

        m1_hat = m1 / (1 - beta1**t)
        m2_hat = m2 / (1 - beta1**t)
        v1_hat = v1 / (1 - beta2**t)
        v2_hat = v2 / (1 - beta2**t)

        w1 -= lr * m1_hat / (math.sqrt(v1_hat) + eps)
        w2 -= lr * m2_hat / (math.sqrt(v2_hat) + eps)

    return steps, loss_and_grad(w1, w2)[0]


def main() -> None:
    max_steps = 500
    lr = 0.1
    threshold = 1e-4

    results = [
        ("SGD", run_sgd(max_steps, lr)),
        ("SGD + Momentum", run_momentum(max_steps, lr)),
        ("Adam", run_adam(max_steps, lr)),
    ]

    print("二维峡谷损失 J = 0.1*w1^2 + w2^2，起点 (5, 5)，lr=0.1")
    print(f"收敛判据：损失 < {threshold}\n")
    print(f"{'优化器':<18} | {'收敛步数':>8} | {'最终损失':>12}")
    print("-" * 44)
    for name, (steps, final_loss) in results:
        status = f"{steps}" if final_loss < threshold else f">{max_steps}"
        print(f"{name:<18} | {status:>8} | {final_loss:>12.6f}")

    print("\n结论：在 w2 更陡的峡谷面上，Momentum / Adam 通常比纯 SGD 更快到达谷底。")


if __name__ == "__main__":
    main()
