"""
极简决策树 + 模拟随机森林投票。

对应文档：docs/02-机器学习基础/005-决策树与集成方法.md
规则与文档手算案例一致：收入(万) <= 5 -> 不买，否则买。

运行：python code/02-机器学习基础/005-决策树与集成方法/decision_tree_demo.py
（仅依赖标准库）
"""

# (年龄标签, 收入万, 是否购买)
SAMPLES = [
    ("青年", 3, False),
    ("青年", 8, True),
    ("中年", 8, True),
]


def gini(labels: list[bool]) -> float:
    """二分类基尼不纯度。"""
    if not labels:
        return 0.0
    p = sum(labels) / len(labels)
    return 1.0 - p * p - (1 - p) * (1 - p)


def predict_single_tree(income: float) -> bool:
    """单棵树：收入 <= 5 不买，否则买。"""
    return income > 5.0


def predict_forest(income: float, n_trees: int = 5) -> bool:
    """
    模拟随机森林多数投票：每棵树在阈值附近加入微小扰动，
    体现 Bagging 降低单棵树方差的思想（演示用）。
    """
    votes = []
    for i in range(n_trees):
        # 扰动阈值模拟不同 bootstrap 样本学到的略有差异的树
        threshold = 5.0 + (i - n_trees // 2) * 0.3
        votes.append(income > threshold)
    return sum(votes) > len(votes) / 2


def main() -> None:
    print("决策树与集成方法演示\n")
    print("训练样本：")
    for age, inc, buy in SAMPLES:
        print(f"  年龄={age}, 收入={inc}万 -> {'买' if buy else '不买'}")

    root_labels = [s[2] for s in SAMPLES]
    print(f"\n根节点基尼不纯度 G = {gini(root_labels):.3f}")
    print('单棵树规则：收入 <= 5 万 -> 不买，否则买\n')

    print("样本 | 单树预测 | 森林投票 | 真实")
    print("-----+----------+----------+------")
    for age, inc, buy in SAMPLES:
        t = predict_single_tree(inc)
        f = predict_forest(inc)
        print(
            f" {inc}万 | {'买' if t else '不买':^8} | {'买' if f else '不买':^8} | {'买' if buy else '不买'}"
        )

    # 边界样本
    edge = 5.0
    print(
        f"\n边界 income={edge} 万：单树={'买' if predict_single_tree(edge) else '不买'}，"
        f"森林={'买' if predict_forest(edge) else '不买'}（阈值扰动时投票更稳）"
    )


if __name__ == "__main__":
    main()
