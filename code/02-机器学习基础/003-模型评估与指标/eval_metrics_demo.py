"""
混淆矩阵、精确率/召回率/F1 与 K 折交叉验证演示。

对应文档：docs/02-机器学习基础/003-模型评估与指标.md
1. 展示类别不平衡下"全判负类"准确率虚高、召回率为 0 的陷阱
2. 演示 5 折交叉验证求平均准确率

运行：python code/02-机器学习基础/003-模型评估与指标/eval_metrics_demo.py
（仅依赖标准库）
"""

import random

# ---------------------------------------------------------------------------
# 第一部分：混淆矩阵与指标（文档案例：1000 人，10 病 990 健康，全判健康）
# ---------------------------------------------------------------------------

TP = 0   # 真阳性：有病且判有病
FP = 0   # 假阳性：没病却判有病
TN = 990
FN = 10


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_confusion_case(title: str, tp: int, fp: int, tn: int, fn: int) -> None:
    m = metrics(tp, fp, tn, fn)
    print(title)
    print(f"  混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  准确率={m['accuracy']:.1%}  精确率={m['precision']:.1%}  "
          f"召回率={m['recall']:.1%}  F1={m['f1']:.3f}")
    print()


# ---------------------------------------------------------------------------
# 第二部分：5 折交叉验证（极简二分类：特征 x，标签 y=1 if x>0.5）
# ---------------------------------------------------------------------------

def simple_predict(x: float) -> int:
    """极简规则分类器：x > 0.5 判正类。"""
    return 1 if x > 0.5 else 0


def k_fold_cv(samples: list[tuple[float, int]], k: int = 5) -> float:
    """K 折交叉验证：每折用 1/k 验证，其余训练（此处规则固定，仅演示流程）。"""
    n = len(samples)
    fold_size = n // k
    scores: list[float] = []

    for fold in range(k):
        start = fold * fold_size
        end = n if fold == k - 1 else start + fold_size
        val = samples[start:end]
        correct = sum(1 for x, y in val if simple_predict(x) == y)
        acc = correct / len(val) if val else 0.0
        scores.append(acc)
        print(f"  折 {fold + 1}/{k}: 验证样本 {len(val)}，准确率 {acc:.1%}")

    return sum(scores) / len(scores)


def main() -> None:
    print("=" * 50)
    print("1. 类别不平衡下的准确率陷阱")
    print("=" * 50)
    print_confusion_case(
        "模型 A：全判健康（文档案例）",
        TP, FP, TN, FN,
    )

    print_confusion_case(
        "模型 B：查出 8 个病人，误报 20 个",
        tp=8, fp=20, tn=970, fn=2,
    )

    print("=" * 50)
    print("2. 5 折交叉验证（20 个样本，规则 x>0.5 判正类）")
    print("=" * 50)
    random.seed(42)
    samples = [(random.random(), 1 if random.random() > 0.4 else 0) for _ in range(20)]
    mean_acc = k_fold_cv(samples, k=5)
    print(f"\n5 折平均准确率：{mean_acc:.1%}")


if __name__ == "__main__":
    main()
