"""
一维 Conditional Flow Matching：线性路径 + Euler ODE 积分。

对应文档：docs/10-生成式AI/006-Flow-Matching基础.md
真实速度 u = x1 - x0；演示从噪声 x0 积分到数据 x1。

运行：python code/10-生成式AI/006-Flow-Matching基础/flow_matching_1d.py
（仅依赖标准库）
"""


def euler_integrate(x_start: float, velocity: float, t_start: float, t_end: float, steps: int) -> float:
    """Euler 法积分 dx/dt = velocity（常数场玩具情形）。"""
    dt = (t_end - t_start) / steps
    x = x_start
    t = t_start
    for _ in range(steps):
        x = x + velocity * dt
        t = t + dt
    return x


def main() -> None:
    x0 = -1.0  # 噪声端
    x1 = 2.0   # 数据端
    true_velocity = x1 - x0  # 线性 CFM：恒为 3

    print("一维 Flow Matching（线性路径，真实速度 u = x1 - x0）")
    print(f"x0={x0}, x1={x1}, u={true_velocity}\n")
    print(f"{'步数 N':>8} | {'积分结果 x(1)':>14} | {'误差':>8}")
    print("-" * 38)

    for n in [1, 2, 4, 8, 16]:
        x_final = euler_integrate(x0, true_velocity, 0.0, 1.0, n)
        err = abs(x_final - x1)
        print(f"{n:>8} | {x_final:>14.4f} | {err:>8.6f}")

    print(f"\n目标 x1={x1}；N=1 时一步可达（完美常数速度场）")


if __name__ == "__main__":
    main()
