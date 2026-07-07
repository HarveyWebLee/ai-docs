"""
多 LoRA 权重线性合并示意（小矩阵）。

对应文档：docs/10-生成式AI/008-多LoRA合并与推理实践.md
W_eff = W0 + w1*s1*B1@A1 + w2*s2*B2@A2

运行：python code/10-生成式AI/008-多LoRA合并与推理实践/multi_lora_merge_demo.py
（仅依赖标准库）
"""


def mat_mul(B: list[list[float]], A: list[list[float]]) -> list[list[float]]:
    """B: d x r, A: r x k -> d x k"""
    d, r = len(B), len(B[0])
    k = len(A[0])
    out = [[0.0] * k for _ in range(d)]
    for i in range(d):
        for j in range(k):
            for t in range(r):
                out[i][j] += B[i][t] * A[t][j]
    return out


def add_scaled(W: list[list[float]], delta: list[list[float]], scale: float) -> None:
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] += scale * delta[i][j]


def frobenius_norm(M: list[list[float]]) -> float:
    return sum(x * x for row in M for x in row) ** 0.5


def main() -> None:
    # 底模权重 W0 (2x2)
    W0 = [[1.0, 0.0], [0.0, 1.0]]

    # LoRA 1: 偏「增大 W[0][0]」
    B1 = [[0.5], [0.0]]
    A1 = [[0.2, 0.0]]
    # LoRA 2: 偏「增大 W[1][1]」
    B2 = [[0.0], [0.5]]
    A2 = [[0.0, 0.2]]

    r1, r2 = 1, 1
    alpha1, alpha2 = 1.0, 1.0
    w1, w2 = 0.8, 0.6

    d1 = mat_mul(B1, A1)
    d2 = mat_mul(B2, A2)
    s1, s2 = alpha1 / r1, alpha2 / r2

    W_dyn = [row[:] for row in W0]
    add_scaled(W_dyn, d1, w1 * s1)
    add_scaled(W_dyn, d2, w2 * s2)

    W_merge = [row[:] for row in W0]
    d_total = [[w1 * s1 * d1[i][j] + w2 * s2 * d2[i][j] for j in range(2)] for i in range(2)]
    add_scaled(W_merge, d_total, 1.0)

    print("多 LoRA 线性合并示意 (2x2 矩阵)\n")
    print("W0 =", W0)
    print(f"LoRA1 weight={w1}, LoRA2 weight={w2}\n")
    print("动态叠加 W_eff =", [[round(x, 4) for x in row] for row in W_dyn])
    print("离线合并 W_eff =", [[round(x, 4) for x in row] for row in W_merge])

    diff = frobenius_norm([[W_dyn[i][j] - W_merge[i][j] for j in range(2)] for i in range(2)])
    print(f"\n两种方式的 Frobenius 差: {diff:.6f} (应≈0)")


if __name__ == "__main__":
    main()
