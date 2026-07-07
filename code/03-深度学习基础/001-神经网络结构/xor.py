"""
XOR（异或）问题：用一个隐层的 ReLU 网络实现线性不可分问题的求解。

对应文档：docs/03-深度学习基础/001-神经网络结构.md 的「案例解析」。
本脚本使用文档中给出的手工权重，逐样本验证输出与真值表一致，
以此证明"隐层引入的非线性组合可解决线性不可分问题"。

运行：python code/03-深度学习基础/001-神经网络结构/xor.py
（仅依赖标准库，无需安装第三方包）
"""


def relu(x: float) -> float:
    """ReLU 激活：小于 0 取 0，否则取原值。"""
    return max(0.0, x)


def xor_network(x1: float, x2: float) -> float:
    """文档中给出的两层网络（隐层 2 个 ReLU 神经元，输出层线性组合）。"""
    # 隐层：h1 捕捉"至少一个为 1"，h2 捕捉"两个都为 1"
    h1 = relu(x1 + x2 - 0.0)
    h2 = relu(x1 + x2 - 1.0)
    # 输出层：y = h1 - 2*h2
    return h1 - 2.0 * h2


def main() -> None:
    truth_table = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0,
    }

    print("x1 x2 | 预测 | 期望 | 是否一致")
    all_ok = True
    for (x1, x2), expected in truth_table.items():
        y = xor_network(x1, x2)
        ok = abs(y - expected) < 1e-9
        all_ok = all_ok and ok
        mark = "OK" if ok else "FAIL"
        print(f" {x1}  {x2} |  {y:.0f}  |  {expected}  | {mark}")

    print("\n结论：", "全部一致，XOR 求解成功。" if all_ok else "存在不一致，请检查权重。")


if __name__ == "__main__":
    main()
