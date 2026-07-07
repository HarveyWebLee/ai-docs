"""
一维线性回归：用梯度下降拟合 y ≈ w*x + b。

对应文档：docs/02-机器学习基础/002-线性回归与逻辑回归.md
数据为人工生成，近似 y = 2x + 1 加少量噪声。

运行：python code/02-机器学习基础/002-线性回归与逻辑回归/linear_regression.py
（仅依赖标准库）
"""

# (面积/特征 x, 房价/标签 y) —— 简化的一维样本
DATA = [
    (1.0, 3.1),
    (2.0, 4.9),
    (3.0, 7.2),
    (4.0, 8.8),
    (5.0, 11.2),
]


def mse_loss(weight: float, bias: float) -> float:
    """均方误差损失。"""
    total = 0.0
    for x, y in DATA:
        pred = weight * x + bias
        total += (pred - y) ** 2
    return total / len(DATA)


def gradient(weight: float, bias: float) -> tuple[float, float]:
    """对 w、b 求 MSE 梯度。"""
    n = len(DATA)
    gw = gb = 0.0
    for x, y in DATA:
        pred = weight * x + bias
        err = pred - y
        gw += err * x
        gb += err
    return 2 * gw / n, 2 * gb / n


def main() -> None:
    weight, bias = 0.0, 0.0
    lr = 0.05
    steps = 200

    print("一维线性回归（梯度下降，MSE 损失）")
    print(f"样本数 {len(DATA)}，学习率 {lr}，迭代 {steps} 步\n")

    for step in range(steps):
        loss = mse_loss(weight, bias)
        if step % 40 == 0 or step == steps - 1:
            print(f"step {step:3d} | w={weight:6.3f} b={bias:6.3f} | MSE={loss:.4f}")
        gw, gb = gradient(weight, bias)
        weight -= lr * gw
        bias -= lr * gb

    print(f"\n拟合结果：y_hat = {weight:.3f} * x + {bias:.3f}")
    print("（真实关系近似 y = 2x + 1）")
    x_test = 6.0
    print(f"预测 x={x_test} -> y_hat={weight * x_test + bias:.2f}")


if __name__ == "__main__":
    main()
