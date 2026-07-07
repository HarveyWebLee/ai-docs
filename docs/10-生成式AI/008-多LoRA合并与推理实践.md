# 008 · 多 LoRA 合并与推理实践

> 本文用大白话回答：ComfyUI 里叠三个 LoRA 各自调 0.8、0.5 是什么意思？能不能把 LoRA「焊死」进底模？多风格一起用为什么经常翻车？
>
> 读完你会知道：多 LoRA = **多个低秩补丁按权重相加**；推理可动态加载，合并可离线算进 $W_0$，但要懂冲突与顺序。

## 一、一句话先说清

**多 LoRA 推理 = 在同一冻结底模 $W_0$ 上，同时加载多个 LoRA 补丁 $(B_i,A_i)$，按权重 $w_i$ 叠加生效：**

$$
W_{\text{eff}} = W_0 + \sum_i w_i \cdot \frac{\alpha_i}{r_i} B_i A_i
$$

- **动态加载**（WebUI / ComfyUI）：推理时挂多个 `.safetensors`，各调 `weight`；
- **离线合并（merge）**：把上式算进 $W_0$，导出**单文件** checkpoint，推理无 LoRA 加载开销；
- **注意**：线性叠加是工程近似，多风格强冲突时质量会下降。

基础概念见 [005 · LoRA 低秩微调](./005-LoRA低秩微调.md)。

## 二、打个比方：底图上贴多张半透明贴纸

[005](./005-LoRA低秩微调.md) 里 LoRA 是**一张贴纸** $BA$。多 LoRA 就像：

| 操作 | 类比 | 工程对应 |
| --- | --- | --- |
| 贴一张贴纸，透明度 80% | 单 LoRA weight=0.8 | 只加载 `style_lora` |
| 叠角色 + 画风两张 | 两 LoRA 同时生效 | `character` + `anime_style` |
| 把贴纸永久压进底图 | 合并进 $W_0$ | `merge_lora` 导出 ckpt |
| 贴纸互相打架 | 风格冲突 | 脸崩、色偏、细节糊 |

**权重 $w_i$** 像贴纸透明度：0 等于没贴，1 为满强度，>1 可能过冲失真。

## 三、它到底解决什么问题

### 问题 1：只想「角色 + 画风 + 概念」组合怎么办

单 LoRA 通常只训一种偏移（某角色或某画风）。实际需求常要：

- 角色 LoRA（脸/服装一致）
- 风格 LoRA（水彩/赛博朋克）
- 可选：Pose / ControlNet（结构，见 [004](./004-ControlNet与可控生成.md)）

**朴素做法**：全参数再训一个「大杂烩」模型——贵且每换组合重来。

**更好做法**：底模 + 多个 LoRA 按需组合，**只调权重**快速试错。

### 问题 2：推理时加载 vs 离线合并

| 方式 | 优点 | 缺点 |
| --- | --- | --- |
| **动态多 LoRA** | 灵活换组合、只占小文件 | 每层多一次 $BAx$；框架要支持 |
| **离线 merge** | 推理简单、兼容老 pipeline | 换组合需重新 merge；冲突固化 |
| **合并后再量化** | 部署省显存 | merge 顺序/权重需先调优 |

常见工具：Kohya `merge.py`、AUTOMATIC1111 「Additional Networks」、ComfyUI `LoraLoader` 链式节点。

### 问题 3：合并公式与顺序

对**同一层**同一 $W_0$，多个 LoRA 增量线性相加：

$$
\Delta W_{\text{total}} = \sum_i w_i \cdot s_i \cdot B_i A_i
$$

其中 $s_i = \alpha_i / r_i$（各 LoRA 训练时的缩放）。

**顺序**：纯线性相加时**理论上可交换**；但 UI 里若还带 bias、block 选择性注入，顺序可能影响数值（实现相关）。**不同层**各挂各的 LoRA 矩阵，由配置文件 `network_module` 指定。

> 对齐：$\sum w_i B_i A_i$ 就是「多张半透明贴纸的效果叠加」——不是矩阵乘法嵌套，是**加法**。

### 问题 4：为什么经常翻车

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| 脸崩/不像 | 角色 LoRA 与风格 LoRA 冲突 | 降低风格 weight；先角色后风格微调权重 |
| 颜色过饱和 | 多个 LoRA weight 叠加 >1 | 各 LoRA 降到 0.5~0.8 |
| 细节糊 | 低秩补丁过多破坏 $W_0$ | 减少同时加载数量（≤3 常见） |
| 与 ControlNet 冲突 | 结构 vs 风格抢梯度 | 降 ControlNet 或 LoRA 其一 |

没有万能权重表——需 **prompt + 种子** 网格搜索；固定 seed 对比 weight。

## 四、专业视角（与大白话对齐）

### 4.1 ComfyUI / A1111 工作流直觉

```
Load Checkpoint (W0)
  → LoraLoader (lora_A, strength=0.8)
  → LoraLoader (lora_B, strength=0.6)
  → KSampler → VAE Decode
```

ComfyUI 将多个 LoRA 解析为对 U-Net / Text Encoder 对应层的 $\Delta W$ 累加。Text Encoder LoRA 影响** prompt 理解**，U-Net LoRA 影响**像素/latent 风格**。

### 4.2 离线合并伪代码

```python
# 概念示意：合并一层权重
W_eff = W0.clone()
for lora in loras:
    W_eff += lora.weight * (lora.alpha / lora.r) * (lora.B @ lora.A)
save_checkpoint(W_eff)
```

合并后原 LoRA 文件不再需要，但**无法再单独调** $w_i$，除非保留未合并副本。

### 4.3 LLM 多 LoRA（了解）

LLM 推理框架（vLLM、S-LoRA）支持 batch 内不同请求挂不同 LoRA，或有限多 adapter；与 SD 「视觉叠加」场景类似，均为 $W_0 + \Delta W_i$，见 [05/003](../05-大语言模型与Transformer/003-预训练与微调.md)。

## 五、案例解析：双 LoRA 权重网格

假设底模 SD 1.5，加载：

- `character_lora`：固定虚拟角色
- `watercolor_lora`：水彩画风

固定 prompt 与 seed=42，扫权重：

| character | watercolor | 直觉效果 |
| --- | --- | --- |
| 1.0 | 0.0 | 角色准，风格像底模 |
| 0.8 | 0.5 | 常见甜点：角色可辨 + 轻度水彩 |
| 0.5 | 1.0 | 水彩强，角色易走样 |
| 1.0 | 1.0 | 易过冲、脸崩 |

工程上先定 character（0.7~0.9），再从小往加 watercolor（0.3→0.6）。

可运行 demo（矩阵叠加示意）：`code/10-生成式AI/008-多LoRA合并与推理实践/multi_lora_merge_demo.py`

```bash
python code/10-生成式AI/008-多LoRA合并与推理实践/multi_lora_merge_demo.py
```

## 六、常见误区与边界

- **误区："merge 一定比动态加载好"**：merge 省加载时间，但失去灵活；开发调参阶段用动态，定稿再 merge 部署。
- **误区："LoRA 可以无限叠"**：有效补丁叠加过多，低秩假设失效，$W_{\text{eff}}$ 偏离合理流形。
- **误区："合并顺序永远无关"**：标准线性相加无关；若含非线性后处理或部分层跳过，以实现为准。
- **边界**：LyCORIS、LoHa 等变体合并规则略异；视频多 LoRA 生态仍在演进（见 [007 视频生成](./007-视频生成基础.md)）。

## 七、一句话总结

- 多 LoRA：$W_{\text{eff}} = W_0 + \sum w_i \frac{\alpha_i}{r_i} B_i A_i$；动态加载灵活，离线 merge 便于部署。
- 控制各 LoRA weight、减少冲突、固定 seed 调参；常与 ControlNet 分工（结构 vs 风格）。
- 上一篇：[007 · 视频生成基础](./007-视频生成基础.md)；基础：[005 · LoRA](./005-LoRA低秩微调.md)；部署：[08/003](../08-AI工程与MLOps/003-模型部署与推理服务.md)。
