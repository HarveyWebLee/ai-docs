"""
8x8 简化"数字"二分类 MLP 训练 demo（0 vs 1）。

对应文档：docs/03-深度学习基础/001-神经网络结构.md
用硬编码的 8x8 像素模式模拟 MNIST 风格图像，训练两层 MLP（ReLU + Sigmoid），
演示"从像素到分类"的完整训练闭环（前向、反向、梯度下降）。

运行：python code/03-深度学习基础/001-神经网络结构/mnist_like_mlp.py
（仅依赖标准库）
"""

import math
import random

# 固定随机种子，便于复现
random.seed(42)

# 8x8 模式：1 表示"笔迹"，0 表示背景（简化版数字 0 与 1）
PATTERN_0 = [
    "01111110",
    "11000011",
    "10000001",
    "10000001",
    "10000001",
    "10000001",
    "11000011",
    "01111110",
]
PATTERN_1 = [
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
]


def pattern_to_vector(rows: list[str]) -> list[float]:
    return [1.0 if c == "1" else 0.0 for row in rows for c in row]


def make_dataset(n_per_class: int = 8) -> list[tuple[list[float], float]]:
    """每类生成 n 个带随机翻转噪声的样本。"""
    base0 = pattern_to_vector(PATTERN_0)
    base1 = pattern_to_vector(PATTERN_1)
    data: list[tuple[list[float], float]] = []
    for _ in range(n_per_class):
        x0 = [v if random.random() > 0.05 else 1.0 - v for v in base0]
        x1 = [v if random.random() > 0.05 else 1.0 - v for v in base1]
        data.append((x0, 0.0))
        data.append((x1, 1.0))
    random.shuffle(data)
    return data


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mat_vec(W: list[list[float]], x: list[float]) -> list[float]:
    return [dot(row, x) for row in W]


def relu(v: float) -> float:
    return max(0.0, v)


def sigmoid(v: float) -> float:
    v = max(-20.0, min(20.0, v))
    return 1.0 / (1.0 + math.exp(-v))


def forward(
    x: list[float],
    W1: list[list[float]],
    b1: list[float],
    W2: list[float],
    b2: float,
) -> tuple[float, list[float], list[float]]:
    """返回 (预测概率, 隐层激活, 隐层 pre-activation)。"""
    z1 = [mat_vec(W1, x)[i] + b1[i] for i in range(len(b1))]
    h = [relu(z) for z in z1]
    z2 = dot(W2, h) + b2
    y_hat = sigmoid(z2)
    return y_hat, h, z1


def train(
    data: list[tuple[list[float], float]],
    hidden: int = 16,
    lr: float = 0.5,
    epochs: int = 200,
) -> tuple[list[list[float]], list[float], list[float], float]:
    dim_in = len(data[0][0])
    # Xavier 风格小随机初始化
    W1 = [[random.uniform(-0.2, 0.2) for _ in range(dim_in)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [random.uniform(-0.2, 0.2) for _ in range(hidden)]
    b2 = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in data:
            y_hat, h, z1 = forward(x, W1, b1, W2, b2)
            # 二元交叉熵
            eps = 1e-7
            loss = -(y * math.log(y_hat + eps) + (1 - y) * math.log(1 - y_hat + eps))
            total_loss += loss

            # 输出层梯度
            d_z2 = y_hat - y
            d_W2 = [d_z2 * h[i] for i in range(hidden)]
            d_b2 = d_z2

            # 隐层梯度（ReLU）
            d_h = [d_z2 * W2[i] for i in range(hidden)]
            d_z1 = [d_h[i] if z1[i] > 0 else 0.0 for i in range(hidden)]

            d_W1 = [[d_z1[i] * x[j] for j in range(dim_in)] for i in range(hidden)]
            d_b1 = d_z1[:]

            W2 = [W2[i] - lr * d_W2[i] for i in range(hidden)]
            b2 -= lr * d_b2
            for i in range(hidden):
                b1[i] -= lr * d_b1[i]
                for j in range(dim_in):
                    W1[i][j] -= lr * d_W1[i][j]

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} | avg BCE = {total_loss / len(data):.4f}")

    return W1, b1, W2, b2


def accuracy(
    data: list[tuple[list[float], float]],
    W1: list[list[float]],
    b1: list[float],
    W2: list[float],
    b2: float,
) -> float:
    correct = 0
    for x, y in data:
        y_hat, _, _ = forward(x, W1, b1, W2, b2)
        pred = 1.0 if y_hat >= 0.5 else 0.0
        if pred == y:
            correct += 1
    return correct / len(data)


def main() -> None:
    train_data = make_dataset(n_per_class=8)
    test_data = make_dataset(n_per_class=4)

    print("8x8 简化数字 MLP（0 vs 1），隐层 16，ReLU + Sigmoid")
    print(f"训练样本 {len(train_data)}，测试样本 {len(test_data)}\n")

    params = train(train_data)
    acc = accuracy(test_data, *params)
    print(f"\n测试集准确率：{acc * 100:.1f}%")
    print("（真实 MNIST 用 28x28 + 更大网络；此处为教学用极简版）")


if __name__ == "__main__":
    main()
