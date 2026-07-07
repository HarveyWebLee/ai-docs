"""
单神经元梯度下降：复现 docs/03-深度学习基础/002-反向传播与梯度下降.md 的手算案例，
并迭代多步直至损失足够小。

模型：y_hat = w * x + b（无激活）
损失：L = 0.5 * (y_hat - y)^2

运行：python code/03-深度学习基础/002-反向传播与梯度下降/gradient_descent.py
（仅依赖标准库）
"""

# 训练样本：x=2 时希望输出 y=4
X = 2.0
Y = 4.0


def forward(x: float, weight: float, bias: float) -> tuple[float, float]:
    """前向：返回预测值 y_hat 与损失 L。"""
    y_hat = weight * x + bias
    loss = 0.5 * (y_hat - Y) ** 2
    return y_hat, loss


def backward(x: float, y_hat: float) -> tuple[float, float]:
    """反向：对 w、b 求损失梯度（链式法则）。"""
    d_loss_d_yhat = y_hat - Y
    d_loss_d_w = d_loss_d_yhat * x
    d_loss_d_b = d_loss_d_yhat
    return d_loss_d_w, d_loss_d_b


def main() -> None:
    # 局部变量，避免与模块级 w/b 混淆
    weight, bias = 0.0, 0.0
    learning_rate = 0.1

    print("单神经元梯度下降（对应 002 文档手算案例）")
    print("x=2, y=4, 初始 w=0, b=0, lr=0.1\n")
    print("step |   w    |   b    | y_hat  |  loss")
    print("-----+--------+--------+--------+--------")

    for step in range(6):
        y_hat, loss = forward(X, weight, bias)
        print(f" {step:3d} | {weight:6.3f} | {bias:6.3f} | {y_hat:6.3f} | {loss:6.3f}")

        if loss < 1e-6:
            print("\n损失已足够小，收敛。")
            break

        gw, gb = backward(X, y_hat)
        weight -= learning_rate * gw
        bias -= learning_rate * gb
    else:
        y_hat, loss = forward(X, weight, bias)
        print(f" {6:3d} | {weight:6.3f} | {bias:6.3f} | {y_hat:6.3f} | {loss:6.3f}")
        print(f"\n6 步后：w={weight:.4f}, b={bias:.4f}，预测 y_hat={y_hat:.4f}（目标 4.0）")


if __name__ == "__main__":
    main()
