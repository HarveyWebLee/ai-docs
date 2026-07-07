"""
一维玩具 GAN 演示：生成器把均匀噪声映射到真实数据附近（0 附近）。

对应文档：docs/10-生成式AI/002-生成对抗网络GAN基础.md
不训练神经网络，用显式规则模拟「G 往 0 靠拢、D 判距 0 的距离」的对抗直觉。

运行：python code/10-生成式AI/002-生成对抗网络GAN基础/gan_1d_demo.py
（仅依赖标准库）
"""

import random

# 真实数据：0 附近（简化为标量 0）
TRUE_MEAN = 0.0
# 生成器状态：当前输出均值（初始远离真实分布）
gen_mean = 2.0
# 判别器阈值：|x| < threshold 判为真
disc_threshold = 1.0


def discriminator(x: float) -> float:
    """输出「像真样本」的分数 [0,1]。"""
    return 1.0 if abs(x - TRUE_MEAN) < disc_threshold else 0.0


def sample_real() -> float:
    """从真实分布采样（0 + 小噪声）。"""
    return random.gauss(TRUE_MEAN, 0.3)


def sample_fake() -> float:
    """生成器输出（当前 gen_mean + 噪声）。"""
    return random.gauss(gen_mean, 0.3)


def main() -> None:
    global gen_mean, disc_threshold

    random.seed(42)
    steps = 8
    lr_g = 0.25
    lr_d = 0.15

    print("一维玩具 GAN（模拟 G 与 D 的拉扯）")
    print(f"真实数据中心={TRUE_MEAN}，初始 G 均值={gen_mean}\n")
    print("step | G均值  | D阈值  | D(真) | D(假) | 说明")
    print("-----+--------+--------+-------+-------+------")

    for step in range(steps + 1):
        x_real = sample_real()
        x_fake = sample_fake()
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake)
        note = ""
        if step < steps:
            # D：缩小阈值，更严格（若假样本能骗过则收紧）
            if d_fake > 0.5:
                disc_threshold = max(0.2, disc_threshold - lr_d)
                note = "D 收紧"
            # G：往真实中心移动
            gen_mean = gen_mean + lr_g * (TRUE_MEAN - gen_mean)
            if abs(gen_mean - TRUE_MEAN) < 0.05:
                note = "G 接近真实"
        print(
            f" {step:3d} | {gen_mean:6.3f} | {disc_threshold:6.3f} | "
            f" {d_real:.0f}    |  {d_fake:.0f}    | {note}"
        )

    print(f"\n最终 G 均值 {gen_mean:.3f}，接近真实 {TRUE_MEAN}（对抗平衡示意）")


if __name__ == "__main__":
    main()
