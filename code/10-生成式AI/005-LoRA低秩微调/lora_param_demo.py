"""
LoRA 可训练参数量 vs 全量微调对比。

对应文档：docs/10-生成式AI/005-LoRA低秩微调.md

运行：python code/10-生成式AI/005-LoRA低秩微调/lora_param_demo.py
（仅依赖标准库）
"""


def full_params(d: int, k: int) -> int:
    return d * k


def lora_params(d: int, k: int, r: int) -> int:
    return r * (d + k)


def main() -> None:
    # 模拟 SD Cross-Attention 一层 (d=k=768)
    d, k = 768, 768
    ranks = [4, 8, 16, 32, 64]

    full = full_params(d, k)
    print(f"权重矩阵 W: {d} x {k} = {full:,} 参数（全微调该层）\n")
    print(f"{'秩 r':>6} | {'LoRA 参数':>12} | {'占全量比例':>10}")
    print("-" * 36)

    for r in ranks:
        lp = lora_params(d, k, r)
        ratio = lp / full * 100
        print(f"{r:>6} | {lp:>12,} | {ratio:>9.2f}%")

    # 10 层 attention 投影粗算
    layers = 10
    r = 16
    total_lora = layers * lora_params(d, k, r)
    total_full = layers * full
    print(f"\n若 {layers} 层均注入 LoRA (r={r}):")
    print(f"  LoRA 总参: {total_lora:,}  vs  全微调: {total_full:,}")
    print(f"  比例: {total_lora / total_full * 100:.2f}%")


if __name__ == "__main__":
    main()
