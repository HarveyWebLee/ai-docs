# 005 · 视觉 Transformer（ViT）

> 本文用大白话回答：Transformer 本来是处理文字的，怎么用到图像上？ViT 为什么说"图像也能切成一串词"？它和 CNN 谁更好？
>
> **零基础读法**：先读第一、二节建立"拼图块当单词"画面；第三节手算 $N=196$ 的例子在第五节有完整 walkthrough；第四节公式可后读；跑完 PyTorch demo 对照 shape；第七节自测后再看总结。
>
> 读完你会知道：ViT 把图片切成小块当"视觉单词"，用标准 Transformer 编码器做分类——用大数据预训练时，它能和 CNN 掰手腕，甚至更强。

## 一、一句话先说清

**ViT（Vision Transformer，视觉 Transformer）= 把一张图切成固定大小的小块（patch，图像块），每个块当成一个"视觉单词"，排成序列后喂给 Transformer 编码器，最后用分类头输出类别。**

一句话类比：**CNN 用滑动小窗口（卷积核）局部看；ViT 把图切成拼图块，像读句子一样从左到右（加位置编码）读这些块。**

| 模型 | 怎么看图 | 一句话 |
| --- | --- | --- |
| CNN | 卷积核在小窗口上滑动，逐层扩大感受野 | "放大镜一层层扫" |
| ViT | 切块 → 序列 → 全局自注意力 | "先切成邮票，再通读全文" |

## 二、打个比方：读文章 vs 用放大镜扫图

| 方式 | 怎么看图 | 对应模型 |
| --- | --- | --- |
| **放大镜局部扫描** | 小窗口滑动，逐层抽象边缘→部件→物体 | CNN（见 [002 卷积神经网络](./002-卷积神经网络CNN.md)） |
| **切成段落再通读** | 整图切成 16×16 的小块，每块当一个词，全局自注意力看关系 | ViT |

CNN 的**归纳偏置（inductive bias，"模型天生自带的假设"）**强：默认"近处像素更相关"（局部性）、"平移后模式不变"（平移等变）——适合**中小数据**，不用太多样本也能学。

ViT **偏置弱**，更"通用"，像一张白纸；但需要**大量数据 + 预训练**才能发挥；小数据集上往往不如 CNN 或需强数据增强。**不是 ViT 一定比 CNN 强，而是"数据够大时 ViT 更吃得开 scaling"。**

## 三、它到底解决什么问题

### 问题 1：能否用"一套 Transformer 架构"统一视觉与语言

NLP 里 [05/002 Transformer](../05-大语言模型与Transformer/002-Transformer架构.md) 已证明自注意力威力；研究者希望视觉也复用同一套积木，便于**多模态**（图文一起训，如 CLIP）。

ViT 的核心思路：**尽量少改 Transformer**，只把输入从 token 序列换成 patch 序列——语言里的"词"换成视觉里的"块"。

### 问题 2：图像二维结构怎么变成一维序列

图像本来是 $H \times W$ 的格子，Transformer 吃的是**一维序列**。ViT 的做法：切块。

设输入 RGB 图像 $H \times W \times 3$，patch 边长 $P$，则 patch 个数：

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

每个 patch 展平成长度 $P^2 \cdot C$ 的向量（$C=3$ 为通道数），经线性投影映射到维度 $D$（与 Transformer 隐藏维一致），得到 $N$ 个 **patch embedding（块嵌入，"每个块的向量表示"）**。

> 对齐：$N$ 就是"句子长度"，每个 embedding 就是一个"视觉单词"。$P$ 越大，块越大、块数 $N$ 越少，看得越"粗"。

| patch 大小 $P$ | $224×224$ 图的 $N$ | 权衡 |
| --- | --- | --- |
| 32 | $(224/32)^2 = 49$ | 算得快，细节少 |
| 16 | $14×14 = 196$ | 常用默认 |
| 8 | $28×28 = 784$ | 细节多，注意力 $O(N^2)$ 很贵 |

### 问题 3：块与块的位置信息从哪来

序列模型需要顺序信息——否则"左上角的块"和"右下角的块"在模型眼里只是两个 bag 里的球。ViT 给每个 patch 加**可学习的位置编码（positional encoding）**（与 NLP 类似），再 prepend 一个 **[CLS] token**（classification token，分类标记，专门用来汇总全局信息、最后送分类头）。

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

**[CLS] 从哪来？** 借鉴 BERT：多一个可学习的特殊向量插在序列最前面，经过 $L$ 层自注意力后，它"看过"所有 patch，适合代表整图做分类。检测/分割任务则常改用别的头（见 [004 目标检测与分割](./004-目标检测与图像分割.md)）。

## 四、专业视角（与大白话对齐）

### 4.1 ViT 前向（分类）

1. **Patch Embedding**：
$$
\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{x}_p^1 E; \dots; \mathbf{x}_p^N E] + \mathbf{E}_{\text{pos}}
$$
其中 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 为投影矩阵，$\mathbf{x}_p^i$ 为第 $i$ 个 patch 展平向量，$[\;;\;]$ 表示拼接。

> 对齐：把 CLS 和所有 patch 投影后排成一行，再加上"每块在第几位"的位置信息——就是"句子 + 位置标签"。

2. **L 层 Transformer Encoder**（多头自注意力 + FFN + LayerNorm，见 [05/001 注意力](../05-大语言模型与Transformer/001-注意力机制与自注意力.md)）；

3. **分类头**：取 $\mathbf{z}_L^0$（CLS 对应向量）经 MLP 输出类别 logits（未归一化的分数，越大越像该类）。

自注意力复杂度 $O(N^2)$，$N$ 大时算力高；常见 $P=16$，$224\times224$ 图得 $N=196$。

### 4.2 ViT vs CNN

| 维度 | CNN | ViT |
| --- | --- | --- |
| 局部先验 | 强（卷积、池化） | 弱（全局注意力） |
| 数据需求 | 中小数据集可训 | 通常需大数据预训练（如 JFT-300M、ImageNet-21k） |
| 可扩展性 | 好 | 堆数据/算力时 scaling 表现好 |
| 下游迁移 | 微调最后几层 | 常微调整个编码器或加 adapter |
| 典型强项 | 移动端、小数据、检测分割骨干 | 大规模预训练、多模态图像编码器 |

**Hybrid（混合）** 变体：前几层用 CNN 提特征图，后面接 Transformer，兼顾局部性与全局建模——小数据上常更稳。

### 4.3 后续发展（了解）

- **Swin Transformer**：窗口内自注意力 + 移位，复杂度近线性，更像"分层 CNN + 注意力"；
- **DeiT**：知识蒸馏，用小模型教 ViT，缓解小数据困境；
- **CLIP / 视觉大模型**：图文对比学习，ViT 作图像编码器（见 [10/003 CLIP](../10-生成式AI/003-多模态CLIP基础.md)）。

## 五、案例解析：224×224 图像如何变成 196 个 token

手算一遍（建议对照此表自己验算）：

- 输入：$224 \times 224 \times 3$
- Patch 大小：$16 \times 16$
- Patch 个数：$(224/16)^2 = 14 \times 14 = 196$
- 每个 patch 展平：$16 \times 16 \times 3 = 768$ 维
- 线性投影到 $D=768$（常见 ViT-Base 配置）→ 196 个 token，再加 1 个 CLS → **序列长度 197**
- 经 12 层 Transformer 后，CLS 向量送分类头得到 1000 类 ImageNet logits

**大白话**：一张图被切成 196 张"小邮票"，每张邮票变成一个 768 维的向量；Transformer 让**任意两张邮票直接对话**（自注意力，不只能看邻居），最后 CLS 汇总全场投票出"这是猫还是狗"。

**和 CNN 同一任务的对比直觉**：CNN 第一层可能只看 $3×3$ 邻域；ViT 第一层自注意力理论上每一对 patch 都能互相看——所以小数据时 ViT 容易"学乱"，大数据时这种全局能力反而是优势。

## 六、常见误区与边界

- **误区："ViT 全面取代 CNN"**：在移动端、小数据、检测分割等场景 CNN 及 CNN-Transformer 混合仍主流；ViT 优势在大规模预训练与 scaling。
- **误区："patch 越大越好"**：$P$ 大则 $N$ 小、算得快，但细粒度信息损失；$P$ 小则序列长、算力涨。
- **误区："ViT 不需要预训练"**：原论文在 ImageNet-1k 上从头训 ViT-L 不如 ResNet；**大数据预训练**是关键。
- **边界**：高分辨率图像需 Swin、局部注意力等降低 $O(N^2)$；纯 ViT 做检测/分割需 DETR、Mask2Former 等专用头（见 [004 目标检测与分割](./004-目标检测与图像分割.md)）。

## PyTorch 可运行示例

安装 CPU 版 PyTorch（仅需一次）：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

运行 PyTorch 版示例：

```bash
python code/06-计算机视觉/005-视觉Transformer ViT/demo_torch.py
```

**输出含义**：脚本用 $8×8$ 灰度图、$4×4$ patch 演示（比正文 $224×16$ 小，方便心算）。你会看到：

- `patches shape` 为 `(1, 4, 16)`：1 张图、4 个 patch、每 patch 展平 16 维（$4×4×1$）；
- `embeddings shape` 为 `(1, 4, 8)`：投影到 `embed_dim=8` 后，每个 patch 一个 8 维向量——对应正文"patch → 线性投影 → D 维"那一步；
- 打印的第 1 个 patch 展平值与 embedding 向量，帮助对照"同一块像素经投影后长什么样"。

## 七、读完后你应该能回答

1. ViT 把二维图像变成 Transformer 输入的三步是什么？$N = (H/P)(W/P)$ 里的 $N$ 代表什么？
2. [CLS] token 和位置编码各解决什么问题？
3. ViT 和 CNN 在"归纳偏置"和数据需求上有什么本质区别？为什么说 ViT 不是"全面更强"？
4. $224×224$、$P=16$ 时序列长度是多少（含 CLS）？自注意力复杂度大致随什么增长？

## 八、一句话总结

- ViT = patch 切分 + 线性嵌入 + 位置编码 + 标准 Transformer 编码器 + CLS 分类。
- 弱归纳偏置、强 scaling；大数据预训练下可与 CNN 竞争，小数据常需蒸馏或 hybrid。
- 上一篇：[004 · 目标检测与图像分割](./004-目标检测与图像分割.md)；延伸：[05 · 大语言模型与 Transformer](../05-大语言模型与Transformer/000-分类总览与知识图谱.md)。
