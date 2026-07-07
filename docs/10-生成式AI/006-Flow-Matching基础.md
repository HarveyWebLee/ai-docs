# 006 · Flow Matching 基础

> 本文用大白话回答：扩散模型要一步步去噪很多步，Flow Matching 说的「学向量场、沿流线走」是什么？它和 DDPM 是替代关系还是一家人？
>
> 读完你会知道：Flow Matching = **学一条从噪声到数据的「流动路线」**，生成时像粒子沿着学到的流速场积分过去——往往比传统扩散**步数更少、训练目标更直接**。

## 一、一句话先说清

**Flow Matching（流匹配）= 学习一个随时间变化的向量场 $v_\theta(x,t)$，使得沿该场从简单分布（如高斯噪声）「流动」到数据分布的轨迹，在终点得到真实样本。**

- **扩散（DDPM）**：学**分数/噪声** $\epsilon_\theta$，多步去噪；
- **Flow Matching**：学**速度** $v_\theta$，用 ODE $\frac{dx}{dt}=v_\theta(x,t)$ 从 $t=0$ 积到 $t=1$ 生成。

代表：**Rectified Flow**、Stable Diffusion 3 部分路线、Flux 等新一代模型采用或借鉴 flow-based 思想。

## 二、打个比方：河道里漂到目的地

| 比喻 | Flow Matching 对应 |
| --- | --- |
| 起点是随机漩涡（噪声） | $x_0 \sim p_0$（简单分布） |
| 终点是风景照分布 | $x_1 \sim p_{\text{data}}$ |
| 河道里每一点的水流方向 | 向量场 $v_\theta(x,t)$ |
| 放一片叶子顺流漂 | 从噪声沿 ODE 积分得到样本 |

扩散像「每天去一点噪、走很多小碎步」；Flow Matching 像「学清整条河道，叶子可以**较大步**沿流线漂到终点」。

## 三、它到底解决什么问题

### 问题 1：扩散采样步数多、慢

DDPM 常需 50~1000 步。Flow Matching 常配合**直线路径（rectified flow）** 或更好 ODE 求解器，用 **10~50 步** 达到可接受质量（模型与实现相关）。

| 范式 | 训练目标直觉 | 采样 |
| --- | --- | --- |
| DDPM | 预测噪声 $\epsilon$ | 多步马尔可夫去噪 |
| Flow Matching | 预测速度 $v$ | ODE 积分（可大步） |

### 问题 2：从数据到噪声的路径怎么定——条件流

**Conditional Flow Matching（CFM）** 为每对 $(x_0, x_1)$（噪声样本, 数据样本）构造插值路径，例如线性：

$$
x_t = (1-t)\, x_0 + t\, x_1, \qquad t \in [0,1]
$$

该路径上的**真实速度**（对 $t$ 求导）：

$$
u_t(x_t \mid x_0, x_1) = \frac{d x_t}{dt} = x_1 - x_0
$$

训练目标：让 $v_\theta(x_t, t)$ 去回归 $u_t$：

$$
L = \mathbb{E}_{x_0,x_1,t}\big[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2\big]
$$

> 对齐：$x_t$ 是噪声与数据之间的「中间态」；网络学的是「此刻该往哪个方向流」——就是 $x_1-x_0$ 这条直线的方向（线性路径情形）。

### 问题 3：和 [001 扩散模型](./001-扩散模型Diffusion基础.md) 的关系

- **扩散**：前向 SDE 加噪固定，反向学 score；与 flow 可在数学上通过变换联系（score 与 velocity 可互推）；
- **Flow Matching**：直接学 ODE 向量场，训练常更稳定、路径可设计（直线、最优传输 OT 等）；
- **实践**：SD3、Flux 等将 flow matching 与 Transformer 架构结合，成为扩散之后的**重要演进方向**，而非完全无关的新物种。

## 四、专业视角（与大白话对齐）

### 4.1 生成（采样）

学好后，从 $x \leftarrow x_0 \sim \mathcal{N}(0,I)$ 出发，数值解 ODE：

$$
\frac{dx}{dt} = v_\theta(x, t), \quad t: 0 \to 1
$$

常用 **Euler** 或 **Heun** 等求解器，步长 $\Delta t = 1/N$，$N$ 为采样步数。

### 4.2 Rectified Flow（了解）

通过迭代「拉直」流动路径，使轨迹更接近直线，进一步减少采样步数与曲线偏差——这是「rectified」名字的由来。

### 4.3 条件生成

与扩散类似，加入文本/类别条件 $c$：学 $v_\theta(x,t,c)$；Classifier-Free Guidance 等技巧可迁移。

## 五、案例解析：一维线性 Flow Matching 手算

数据点 $x_1 = 2$，噪声 $x_0 = -1$，线性路径：

$$
x_t = (1-t)(-1) + t \cdot 2 = -1 + 3t
$$

速度恒为 $u = x_1 - x_0 = 3$。

设某步 $t=0.5$，则 $x_{0.5} = 0.5$。若网络已完美学到 $v_\theta \equiv 3$，Euler 一步 $\Delta t=0.5$：

$$
x \leftarrow 0.5 + 3 \times 0.5 = 2 = x_1
$$

**一步即到数据点**（线性路径 + 完美 $v$ 的玩具情形）；高维非线性时需多步积分。

可运行 demo：`code/10-生成式AI/006-Flow-Matching基础/flow_matching_1d.py`

```bash
python code/10-生成式AI/006-Flow-Matching基础/flow_matching_1d.py
```

## 六、常见误区与边界

- **误区："Flow Matching 完全取代扩散"**：二者理论相通；工程上 flow 系模型增多，但 DDPM/DDIM 生态仍庞大。
- **误区："步数一定越少越好"**：步数过少会累积 ODE 离散误差；需与求解器阶数、模型容量权衡。
- **边界**：最优传输 Flow、Stochastic Interpolants 等为活跃研究方向；视频/3D flow 生成仍在演进。

## 七、一句话总结

- Flow Matching 学向量场 $v_\theta$，用 ODE 从噪声流到数据；线性 CFM 目标为回归 $x_1-x_0$。
- 相对经典扩散，常更少采样步、训练形式简洁；SD3/Flux 等代际模型的重要技术之一。
- 上一篇：[005 · LoRA 低秩微调](./005-LoRA低秩微调.md)；下一篇：[007 · 视频生成基础](./007-视频生成基础.md)；多 LoRA 实践：[008 · 多 LoRA 合并与推理](./008-多LoRA合并与推理实践.md)。
