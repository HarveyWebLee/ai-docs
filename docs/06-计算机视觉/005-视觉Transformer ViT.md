# 005 · 视觉 Transformer（ViT）

> 本文用大白话回答：Transformer 本来是处理文字的，怎么用到图像上？ViT 为什么说"图像也能切成一串词"？它和 CNN 谁更好？
>
> 读完你会知道：ViT 把图片切成小块当"视觉单词"，用标准 Transformer 编码器做分类——用大数据预训练时，它能和 CNN 掰手腕，甚至更强。

## 一、一句话先说清

**ViT（Vision Transformer）= 把一张图切成固定大小的小块（patch），每个块当成一个"视觉单词"，排成序列后喂给 Transformer 编码器，最后用分类头输出类别。**

一句话类比：**CNN 用滑动小窗口（卷积核）局部看；ViT 把图切成拼图块，像读句子一样从左到右（加位置编码）读这些块。**

## 二、打个比方：读文章 vs 用放大镜扫图

| 方式 | 怎么看图 | 对应模型 |
| --- | --- | --- |
| **放大镜局部扫描** | 小窗口滑动，逐层抽象边缘→部件→物体 | CNN（见 [002 卷积神经网络](./002-卷积神经网络CNN.md)） |
| **切成段落再通读** | 整图切成 16×16 的小块，每块当一个词，全局自注意力看关系 | ViT |

CNN 的**归纳偏置**强：局部性、平移等变——适合中小数据。ViT **偏置弱**，更"通用"，但需要**大量数据 + 预训练**才能发挥；小数据集上往往不如 CNN 或需强数据增强。

## 三、它到底解决什么问题

### 问题 1：能否用"一套 Transformer 架构"统一视觉与语言

NLP 里 [05/002 Transformer](../05-大语言模型与Transformer/002-Transformer架构.md) 已证明自注意力威力；研究者希望视觉也复用同一套积木，便于**多模态**（图文一起训）。

ViT 的核心思路：**尽量少改 Transformer**，只把输入从 token 序列换成 patch 序列。

### 问题 2：图像二维结构怎么变成一维序列

设输入 RGB 图像 $H \times W \times 3$，patch 边长 $P$，则：

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

个 patch。每个 patch 展平成长度 $P^2 \cdot 3$ 的向量，经线性投影映射到维度 $D$（与 Transformer 隐藏维一致），得到 $N$ 个 **patch embedding**。

> 对齐：$N$ 就是"句子长度"，每个 embedding 就是一个"视觉单词"。

### 问题 3：块与块的位置信息从哪来

序列模型需要顺序信息。ViT 给每个 patch 加**可学习的位置编码**（与 NLP 类似），再 prepend 一个 **[CLS] token**（分类标记，汇总全局信息用于分类）。

整体流程：

```mermaid
graph LR
  Img["图像 H×W"] --> Patch["切 patch ×N"]
  Patch --> Emb["线性投影 → D 维"]
  Emb --> Pos["+ 位置编码 + CLS"]
  Pos --> Enc["Transformer 编码器 ×L"]
  Enc --> Cls["取 CLS 输出 → 分类头"]

  classDef step fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
  class Img,Patch,Emb,Pos,Enc,Cls step;
```

## 四、专业视角（与大白话对齐）

### 4.1 ViT 前向（分类）

1. **Patch Embedding**：$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{x}_p^1 E; \dots; \mathbf{x}_p^N E] + \mathbf{E}_{\text{pos}}$，$E \in \mathbb{R}^{(P^2 \cdot C) \times D}$；
2. **L 层 Transformer Encoder**（多头自注意力 + FFN + LayerNorm，见 [05/001 注意力](../05-大语言模型与Transformer/001-注意力机制与自注意力.md)）；
3. **分类头**：取 $\mathbf{z}_L^0$（CLS 对应向量）经 MLP 输出类别 logits。

自注意力复杂度 $O(N^2)$，$N$ 大时算力高；常见 $P=16$，$224\times224$ 图得 $N=196$。

### 4.2 ViT vs CNN

| 维度 | CNN | ViT |
| --- | --- | --- |
| 局部先验 | 强（卷积、池化） | 弱（全局注意力） |
| 数据需求 | 中小数据集可训 | 通常需大数据预训练（如 JFT-300M、ImageNet-21k） |
| 可扩展性 | 好 | 堆数据/算力时 scaling 表现好 |
| 下游迁移 | 微调最后几层 | 常微调整个编码器或加 adapter |

**Hybrid** 变体：前几层用 CNN 提特征，后面接 Transformer，兼顾局部性与全局建模。

### 4.3 后续发展（了解）

- **Swin Transformer**：窗口内自注意力 + 移位，复杂度近线性，更像"分层 CNN + 注意力"；
- **DeiT**：知识蒸馏，小数据上训 ViT；
- **CLIP / 视觉大模型**：图文对比学习，ViT 作图像编码器。

## 五、案例解析：224×224 图像如何变成 196 个 token

- 输入：$224 \times 224 \times 3$
- Patch 大小：$16 \times 16$
- Patch 个数：$(224/16)^2 = 14 \times 14 = 196$
- 每个 patch 展平：$16 \times 16 \times 3 = 768$ 维
- 线性投影到 $D=768$（常见配置）→ 196 个 token，再加 1 个 CLS → 序列长度 197
- 经 12 层 Transformer 后，CLS 向量送分类头得到 1000 类 ImageNet logits

**大白话**：一张图被切成 196 张"小邮票"，每张邮票变成一个 768 维的向量；Transformer 让任意两张"邮票"直接对话（自注意力），最后 CLS 汇总全场投票出"这是猫还是狗"。

## 六、常见误区与边界

- **误区："ViT 全面取代 CNN"**：在移动端、小数据、检测分割等场景 CNN 及 CNN-Transformer 混合仍主流；ViT 优势在大规模预训练与 scaling。
- **误区："patch 越大越好"**：$P$ 大则 $N$ 小、算得快，但细粒度信息损失；$P$ 小则序列长、算力涨。
- **边界**：高分辨率图像需 Swin、局部注意力等降低 $O(N^2)$；纯 ViT 做检测/分割需 DETR、Mask2Former 等专用头（见 [004 目标检测与分割](./004-目标检测与图像分割.md)）。

## 七、一句话总结

- ViT = patch 切分 + 线性嵌入 + 位置编码 + 标准 Transformer 编码器 + CLS 分类。
- 弱归纳偏置、强 scaling；大数据预训练下可与 CNN 竞争，小数据常需蒸馏或 hybrid。
- 上一篇：[004 · 目标检测与图像分割](./004-目标检测与图像分割.md)；延伸：[05 · 大语言模型与 Transformer](../05-大语言模型与Transformer/000-分类总览与知识图谱.md)。
