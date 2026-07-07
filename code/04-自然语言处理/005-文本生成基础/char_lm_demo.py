"""
字符级 bigram 语言模型 + 贪心解码接龙。

对应文档：docs/04-自然语言处理/005-文本生成基础.md
从训练语料统计 P(下一字 | 当前字)，再给定前缀逐字生成。

运行：python code/04-自然语言处理/005-文本生成基础/char_lm_demo.py
（仅依赖标准库）
"""

CORPUS = [
    "我爱中国",
    "中国人民",
    "中国爱我",
]


def train_bigram(corpus: list[str]) -> dict[str, dict[str, int]]:
    """统计字符 bigram 共现次数：counts[c1][c2]。"""
    counts: dict[str, dict[str, int]] = {}
    for text in corpus:
        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i + 1]
            counts.setdefault(c1, {})
            counts[c1][c2] = counts[c1].get(c2, 0) + 1
    return counts


def next_char(counts: dict[str, dict[str, int]], current: str) -> str | None:
    """贪心：选共现次数最多的下一字。"""
    if current not in counts:
        return None
    followers = counts[current]
    return max(followers, key=followers.get)


def generate(counts: dict[str, dict[str, int]], prefix: str, max_len: int = 20) -> str:
    """自回归接龙：反复 append 下一字直到无法继续或达上限。"""
    result = prefix
    for _ in range(max_len - len(prefix)):
        nxt = next_char(counts, result[-1])
        if nxt is None:
            break
        result += nxt
    return result


def main() -> None:
    counts = train_bigram(CORPUS)
    print("训练语料：", CORPUS)
    print("\n学到的 bigram（部分）：")
    for c1 in sorted(counts):
        pairs = ", ".join(f"{c2}:{n}" for c2, n in sorted(counts[c1].items()))
        print(f"  '{c1}' -> {pairs}")

    prefixes = ["我", "中", "中国"]
    print("\n贪心接龙生成：")
    for p in prefixes:
        out = generate(counts, p)
        print(f"  前缀 '{p}' -> '{out}'")


if __name__ == "__main__":
    main()
