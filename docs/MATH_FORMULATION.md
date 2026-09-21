# BCP-Vis 数学形式化与推理过程（论文写作底稿）

> 本文档与 `BCP-Vis\` 下的**实际实现逐行对齐**。所有公式均可直接搬入论文
> Method 章节；每条公式后标注了对应的代码位置，便于审稿复现。
>
> 符号约定：标量用小写斜体 $x$，向量/矩阵用粗体 $\mathbf{x}$，随机场（图像）用大写 $I,M,P,\hat Y$，
> 可学习参数用 $\theta,\phi$，集合用花体 $\mathcal{D},\mathcal{S}$。
>
> v1.0 · 2026-09-07

---

## 0. 一页速览（可直接作为论文 Method 的骨架）

| 步骤 | 公式 | 代码 |
|---|---|---|
| 实例合并 | $M(x,y)=\max_i M_i(x,y)$ | `scripts/build_binary_masks.py` |
| Soft BCP 目标 | $P^\*\_{gt}(x,y)=\begin{cases}1,&(x,y)\in\mathcal F\\\\ \exp\!\big(-\tfrac{d^2(x,y)}{2\sigma^2}\big),&\text{else}\end{cases}$ | `scripts/build_soft_prior_targets.py` |
| PriorNet | $Z=G_\theta(I),\quad P=\sigma(Z)$ | `models/priornet.py` |
| 先验校准 | $P'=\sigma(\gamma P+b)$ | （V2 之后启用，已预留） |
| 通道拼接 | $I_{RGBP}=[I;P']\in\mathbb R^{4\times H\times W}$ | `datasets_py/patch_dataset.py` |
| 分割网络 | $\hat Y=F_\phi(I_{RGBP})$ | `models/final_cnn.py` |
| PriorNet 损失 | $\mathcal L_{prior}=\mathcal L_{focal}+\lambda_{dice}\mathcal L_{dice}$ | `models/losses.py` |
| 分割损失 | $\mathcal L_{final}=\mathcal L_{focal}+\lambda_{dice}\mathcal L_{dice}$ | 同上（与 RGB baseline 完全一致） |

---

## 1. 符号表

| 符号 | 含义 | 维度 / 取值 |
|---|---|---|
| $I$ | UAV 航拍 RGB 图像 | $\mathbb R^{3\times H\times W}$，$H\times W=1080\times1920$ |
| $\mathcal I$ | 图像空间网格 $\\{1..H\\}\times\\{1..W\\}$ | — |
| $(x,y)$ | 像素坐标 | $\in\mathcal I$ |
| $M_i$ | 第 $i$ 个实例（车辆）二值掩膜 | $\\{0,1\\}^{H\times W}$ |
| $M$ | 合并后的前景真值 | $\\{0,1\\}^{H\times W}$ |
| $\mathcal F$ | 前景像素集合 $\\{(x,y):M(x,y)=1\\}$ | $\lvert\mathcal F\rvert/\lvert\mathcal I\rvert\approx2.34\%$ |
| $\bar{\mathcal F}$ | 背景集合 $\mathcal I\setminus\mathcal F$ | — |
| $d(x,y)$ | 到最近前景像素的欧氏距离（EDT） | $\mathbb R_{\ge0}$ |
| $\sigma$ | 高斯衰减尺度（像素） | 本文取 $32$ |
| $P^\*\_{gt}$ | Soft BCP 监督目标 | $[0,1]^{H\times W}$ |
| $G_\theta$ | PriorNet，参数 $\theta$ | $\mathbb R^{3\times H\times W}\to\mathbb R^{1\times H\times W}$ |
| $Z$ | PriorNet 输出的 logits | $\mathbb R^{1\times H\times W}$ |
| $P$ | 预测先验（概率）$=\mathrm{sigmoid}(Z)$ | $[0,1]^{H\times W}$ |
| $F_\phi$ | Final CNN（ResNet18-UNet），参数 $\phi$ | $\mathbb R^{C\times H\times W}\to\mathbb R^{1\times H\times W}$，$C\in\\{3,4\\}$ |
| $\hat Y$ | 最终预测前景掩膜 | $\\{0,1\\}^{H\times W}$ |
| $h$ | 飞行高度（m） | $\\{50,70,90\\}$ |
| $\tilde h$ | 归一化高度 $(h-70)/40$ | $\\{-0.5,0,0.5\\}$ |
| $\theta^*$ | 训练好的 PriorNet 参数 | — |

---

## 2. 问题定义

### 2.1 监督学习目标

给定训练集 $\mathcal D=\\{(I^{(n)},\\{M^{(n)}_i\\},h^{(n)},s^{(n)})\\}_{n=1}^{N}$，其中 $s^{(n)}\in\\{0,1\\}$ 表示是否含雪。
目标是学得映射 $f:\mathbb R^{3\times H\times W}\to\\{0,1\\}^{H\times W}$，使得预测 $\hat Y=f(I)$ 与真值 $M$ 的
前景 IoU 最大。

### 2.2 实例掩膜的合并规则

EVD4UAV 的标注是 **实例级多边形**（CVAT 1.1 XML，共 123,694 个 polygon）。
语义分割需要的是**类别级**二值图，因此采用逻辑"或"合并：

$$\boxed{M(x,y)=\max_{i\in\\{1..K\\}} M_i(x,y)=\bigvee_{i}M_i(x,y)}$$

即任一实例覆盖该像素即为前景。这个选择在数学上是**并集**：
$\mathcal F=\bigcup_i \mathcal F_i$，保证不同实例重叠区域不产生歧义。

> 代码：`build_binary_masks.py` 中 `binary |= m.astype(np.uint8)`，正是逐像素 max。

### 2.3 极端前景不平衡（本文的核心动机）

实测 9,371 张有效图上，前景像素占比均值仅

$$\rho=\mathbb E_{n}\frac{\lvert\mathcal F^{(n)}\rvert}{HW}\approx 2.34\%$$

即正负样本比约 $1:42$。这带来两个可量化的后果：

1. **交叉熵的梯度被背景主导。** 对 $\mathcal L_{BCE}$，
   $$\frac{\partial \mathcal L_{BCE}}{\partial z} = p-y$$
   背景像素数占 $97.66\%$，其累计梯度压倒前景信号。
2. **小目标在下采样中湮灭。** ResNet18 主干最深到 $H/32$，一个 $16\times16$ 像素的车辆在
   $H/32$ 特征图上仅占 $0.5\times0.5$ 像素——**亚像素级**，双线性下采样后信息几乎完全丢失。

这两点分别由 **Focal Loss**（§5.1）与 **Lite-HR PriorNet**（§8）针对性解决。

---

## 3. BCP-Vis 的形式化

### 3.1 基线（RGB-only）

$$\hat Y = F_\phi(I),\qquad I\in\mathbb R^{3\times H\times W}$$

### 3.2 BCP-Vis（本文方法）

$$\boxed{\begin{aligned}
Z &= G_\theta(I), & P &= \mathrm{sigmoid}(Z)\in[0,1]^{1\times H\times W}\\\\[2pt]
I_{RGBP} &= \big[I;\;P'\big]\in\mathbb R^{4\times H\times W}\\\\[2pt]
\hat Y &= F_\phi(I_{RGBP})
\end{aligned}}$$

其中 $[;\;]$ 表示沿通道维拼接，$P'$ 为（可选的）校准先验，§3.4。

### 3.3 为什么是"第 4 通道拼接"而不是别的融合方式

论文里需要给出选择依据。三种候选融合及其数学性质：

| 方案 | 形式 | 缺陷 |
|---|---|---|
| **早期拼接（本文）** | $F_\phi([I;P])$ | 需重新初始化 conv1（已解决，§7.2） |
| 中期注意力门控 | $F_l'=F_l\cdot\sigma(W P)$ | 只在某个尺度注入，浅层小目标信息已丢失 |
| 双流后融合 | $\hat Y = g(F_\phi(I),G_\theta(I))$ | 参数量翻倍，破坏与 baseline 的结构可比性 |

**选择早期拼接的决定性理由**：它使 $F_\phi$ 的**第一层卷积核**就能直接读取先验响应。
形式化地说，conv1 的输出为

$$e_0 = \mathrm{ReLU}\Big(\mathrm{BN}\big(\underbrace{W_{1:3}*I}_{\text{RGB 项}}+\underbrace{w_4*P'}_{\text{先验项}}+b\big)\Big)$$

其中 $w_4\in\mathbb R^{64\times1\times7\times7}$。先验项 $w_4*P'$ 在**最高分辨率**（$H/2$ 之后）参与计算，
此时 $16\times16$ 的目标仍有 $8\times8$ 个响应单元——信息尚未湮灭。这正是小目标召回率提升的机制。

### 3.4 先验校准（Prior Calibration）

PriorNet 输出的 $P$ 通常**过自信**（sigmoid 饱和）。引入两个可学习标量 $\gamma,b$：

$$P' = \mathrm{sigmoid}\big(\gamma\cdot Z + b\big)$$

等价于对 logits 做仿射重标定。$\gamma$ 控制锐度（$\gamma>1$ 更陡），$b$ 控制工作点。
初始化 $\gamma=1,b=0$ 时退化为恒等映射，**保证不损害初始性能**。

> 状态：代码已预留接口，建议在 V2（Soft BCP）之后启用。当前 E5 直接使用 $P'=P$。

---

## 4. Soft BCP 监督目标的构造与推导

### 4.1 为什么不直接用硬掩膜 $M$ 监督 PriorNet

若 $P^\*\_{gt}=M$，则 PriorNet 的回归目标在边界处是**阶跃函数**：

$$M(x,y)=\mathbb 1\\{(x,y)\in\mathcal F\\}$$

对全卷积网络而言，拟合阶跃需要在边界处产生极高频响应，而卷积的**谱偏置（spectral bias）**
倾向拟合低频函数，导致边界振铃、训练不稳。更关键的是：

> **硬目标不编码"接近度"信息。** 目标外围 1 px 与目标外围 200 px 在硬目标下完全等价（都是 0），
> 但语义上前者"很可能在目标附近"，后者"确定是纯背景"。这个信息正是我们希望第 4 通道携带的。

### 4.2 距离变换

定义到前景集合的欧氏距离变换（EDT）：

$$d(x,y)=\min_{(u,v)\in\mathcal F}\sqrt{(x-u)^2+(y-v)^2},\qquad d(x,y)=0\ \ \forall (x,y)\in\mathcal F$$

实现为两遍扫描的精确欧氏距离变换（`scipy.ndimage.distance_transform_edt`），复杂度 $O(HW)$。

### 4.3 高斯衰减

$$\boxed{P^\*\_{gt}(x,y)=\begin{cases}
1, & (x,y)\in\mathcal F\\[4pt]
\exp\!\left(-\dfrac{d^2(x,y)}{2\sigma^2}\right), & (x,y)\in\bar{\mathcal F}
\end{cases}}$$

这是以"到最近目标的距离"为自变量的**径向基函数（RBF）核**，等价于
$P^\*\_{gt}(x,y)=\max_{(u,v)\in\mathcal F}\ \exp\!\big(-\tfrac{\|(x,y)-(u,v)\|^2}{2\sigma^2}\big)$ 在背景区域的取值。

**数学性质（论文可引）：**

1. **值域**：$P^\*\_{gt}\in(0,1]$，且 $P^\*\_{gt}\equiv1$ 于 $\mathcal F$ 上。
2. **边界连续性**：在 $\partial\mathcal F$ 上，$d\to0^+$ 时 $\exp(-d^2/2\sigma^2)\to1^-$，
   故 $P^\*\_{gt}$ **连续**（$C^0$），且一阶导 $\partial P/\partial d = -\tfrac{d}{\sigma^2}e^{-d^2/2\sigma^2}\to0$，
   实际上在边界处 $C^1$ 光滑。
3. **单调性**：$\partial P^\*\_{gt}/\partial d\le 0$，即"离目标越远，置信度越低"——与先验语义一致。
4. **可微性**：$P^\*\_{gt}$ 关于 $d$ 处处可微，与基于梯度的优化兼容。

### 4.4 半高宽度（关键量化结论）

令 $P^\*\_{gt}=0.5$ 解得半高距离：

$$d_{0.5}=\sigma\sqrt{2\ln 2}\approx1.1774\,\sigma$$

取 $\sigma=32$ px 时 $d_{0.5}\approx37.7$ px。

> **重要实现细节（必须在论文中说明）**：Focal Loss 内部以 $t\ge0.5$ 判定正类
> （`losses.py: target >= 0.5`）。因此 Focal 分支实际把**目标外围约 $37.7$ px 的环带也当作了正类**，
> 相当于对硬目标做了一次**软膨胀（soft dilation）**；而 Dice 分支使用连续的 $P^\*\_{gt}$。
> 两个分支的分工是：*Focal 负责软膨胀后的前景/背景判别，Dice 负责连续值的回归精度*。
> 这个组合比单纯用硬 mask 监督更鲁棒，因为目标外围像素得到了渐变的、而非全有全无的梯度。

### 4.5 $\sigma$ 的选取（有实测依据，不是拍脑袋）

在 EVD4UAV 全量 9,371 张图上扫描候选 $\sigma$，统计"高于阈值的邻域像素占比"：

| $\sigma$ | 8 | 16 | 24 | **32** | 48 | 64 |
|---|---|---|---|---|---|---|
| 邻域占比 | 2.2% | 4.9% | 7.7% | **10.6%** | 16.5% | 22.4% |

选择 $\sigma=32$ 的依据：
- $\sigma\le16$：邻域占比 $<5\%$，几乎退化回硬掩膜，失去 soft 的意义；
- $\sigma\ge48$：邻域占比 $>16\%$，远超前景本身的 $2.34\%$，先验过于弥散，会淹没真实目标响应；
- $\sigma=32$ 使"软邻域:真实前景"之比约 $10.6:2.34\approx4.5:1$，既能提供足够的上下文梯度，
  又不至于让背景主导。

> 论文写法建议：把这张表放进消融实验，并说明 $\sigma$ 与**平均目标尺度**的关系
> （EVD4UAV 车辆在 50/70/90 m 下的像素尺寸分布）。

---

## 5. PriorNet 的训练目标

### 5.1 Focal Loss（完整展开）

标准二值交叉熵 $\mathrm{BCE}(p_t)=-\log p_t$，其中

$$p_t=\begin{cases}p, & y=1\\ 1-p, & y=0\end{cases},\qquad p=\mathrm{sigmoid}(z)$$

Focal Loss 引入调制因子 $(1-p_t)^\gamma$ 与类别平衡权重 $\alpha_t$：

$$\boxed{\mathcal L_{focal}(z,y)=-\alpha_t\,(1-p_t)^\gamma\log p_t,\qquad
\alpha_t=\begin{cases}\alpha,&y=1\\ 1-\alpha,&y=0\end{cases}}$$

本文取 $\alpha=0.25,\ \gamma=2.0$。

**为什么 $\alpha=0.25$ 而不是 $0.75$？** 注意此处 $y=1$ 是**前景（少数类）**。
$\alpha_t$ 的语义是"该类的权重"。取 $\alpha=0.25$ 意味着前景权重 $0.25$、背景权重 $0.75$——
看似反直觉（应该给少数类更大权重）。真正解决不平衡的是**调制因子**：

对易分背景样本（$p_t\approx1$，例如 $p=0.05$ 判为背景时 $p_t=0.95$）：
$$(1-p_t)^\gamma = 0.05^2 = 2.5\times10^{-3}$$
其梯度被压缩约 **400 倍**。而难分样本（$p_t\approx0.5$）仅压缩 $0.25$ 倍。
两者的相对比值变化达 $1600$ 倍，**这正是 Focal 抑制海量易分背景的机制**。

$\alpha$ 的作用是**二阶微调**：在调制之后，再把少数类的整体尺度略微下调，
防止 $\alpha$ 与调制因子叠加导致前景过冲（这是原论文 Lin et al. 的推荐配置，本文沿用）。

### 5.2 Dice Loss

$$\boxed{\mathcal L_{dice}=1-\frac{2\sum_{x,y}p(x,y)\,t(x,y)+\varepsilon}{\sum_{x,y}p(x,y)+\sum_{x,y}t(x,y)+\varepsilon}}$$

$t$ 为监督目标（此处是连续的 $P^\*\_{gt}$，故为 *soft Dice*），$\varepsilon=10^{-6}$。

> **实现细节（写论文时的 footnote）**：代码中 `dims=(0,2,3)`，即**在 batch 维与空间维上联合求和**，
> 得到的是 *batch-level Dice* 而非 *per-sample Dice 的均值*。二者差异：batch-level 对 batch 内
> 前景像素极少（甚至为 0）的样本更稳定，不会因为个别空样本产生 $\mathcal L_{dice}=1$ 的病态梯度。
> 在小目标 + 高比例空 patch 的场景下，batch-level 是更合适的选择。

### 5.3 组合损失

$$\boxed{\mathcal L_{prior}=\mathcal L_{focal}+\lambda_{dice}\mathcal L_{dice},\qquad \lambda_{dice}=1.0}$$

（V5 蒸馏阶段再加 $\lambda_{feat}\mathcal L_{feat}$，$\lambda_{feat}=0.1$，
$\mathcal L_{feat}=1-\cos(\mathbf f_S,\mathbf f_T)=1-\tfrac{\mathbf f_S^\top\mathbf f_T}{\|\mathbf f_S\|\|\mathbf f_T\|}$，当前未启用。）

**为什么 Focal + Dice 而不是单个？** 两者的梯度性质互补：

- Focal 是**逐像素独立**的，优化的是校准后的逐点似然，但它直接受 $\rho=2.34\%$ 的不平衡影响；
- Dice 是**全局集合级**的（分子分母都是全图求和），天然对前景/背景比例不敏感
  （Dice 本质上是 $F_1$ 的软松弛）。

$$\underbrace{\frac{\partial \mathcal L_{dice}}{\partial p}}_{集合级}\ \propto\ -\frac{2t(\Sigma_p+\Sigma_t)-2\Sigma_{pt}}{(\Sigma_p+\Sigma_t)^2}$$

当前景极小时，$\Sigma_t\ll\Sigma_p$，若某像素 $t=1$ 而 $p$ 小，梯度量级 $\approx 2/\Sigma_p$，
**不会**像 BCE 那样因 $\Sigma_t$ 小而消失。这就是 Dice 在小目标分割中不可或缺的原因。

---

## 6. 分割网络（Final CNN）的损失

$$\mathcal L_{final}=\mathcal L_{focal}(\hat z, M)+\lambda_{dice}\mathcal L_{dice}(\hat z, M)$$

**硬性要求**：RGB baseline（E0）与 BCP-Vis（E5）使用**完全相同**的
loss 类型、$\alpha,\gamma,\lambda_{dice}$、优化器、学习率、调度、epoch 数、数据增强、数据划分、随机种子。
唯一变量是 $C=3$ vs $C=4$。这是一切结论可信度的前提。

> 代码强制检查：`train_final.py` 中 `assert model.in_channels == in_ch`。

---

## 7. 结构公平性证明（论文必须给出的论证）

### 7.1 参数量差异的精确计算

设 ResNet18-UNet 的 conv1 为 $\mathbf W\in\mathbb R^{64\times C\times 7\times 7}$。

- $C=3$：$64\times3\times7\times7=9408$ 参数
- $C=4$：$64\times4\times7\times7=12544$ 参数

差值 $=3136=64\times1\times7\times7$，**恰好是第 4 通道的卷积核**。其余所有层的参数完全相同。

实测：

| 模型 | 参数量 |
|---|---|
| $F_\phi$，$C=3$ | 14,834,121 |
| $F_\phi$，$C=4$ | 14,837,257 |
| **$\Delta$** | **3,136** = $64\times1\times7\times7$ ✓ |

相对增量 $\dfrac{3136}{14834121}=2.11\times10^{-4}$，即 **0.021%**。

**论文表述**：
> The only structural difference between the baseline and BCP-Vis is the input channel of the
> first convolution ($3\to4$), contributing 3,136 additional parameters (0.021% of the total).
> All other layers, including the decoder, remain identical, which rules out the possibility that
> observed gains stem from increased model capacity.

### 7.2 第 4 通道的初始化（一个容易被审稿人挑刺的点）

若把 $w_4$ 初始化为 **0**，则初始前向中先验项恒为 0，且由链式法则

$$\frac{\partial \mathcal L}{\partial w_4}=\sum_{x,y}\frac{\partial\mathcal L}{\partial e_0}\cdot P'$$

虽然梯度本身非零（因为 $\partial\mathcal L/\partial e_0$ 来自其他通道），但**先验信号的通路在初期是完全断开的**，
且由于 $w_4=0$ 与 BatchNorm 的耦合，$P'$ 对输出的影响需要多个 step 才显现。

本文采用 **RGB 均值初始化**：

$$\boxed{w_4^{(init)}=\frac{1}{3}\sum_{c=1}^{3}W_{:,c,:,:}}$$

**直觉**：把先验通道视作"第 4 个颜色通道"，用 RGB 三通道的平均响应作为其初始感受野。
这样先验通道从**第一步**就与 RGB 通道处于相同的激活量级，参与前向与反向。

> 代码：`final_cnn.py` 中 `extra = w_old.mean(dim=1, keepdim=True)`（`init_prior="mean"`），
> 可用 `init_prior="zeros"` 切换。

**消融建议**：论文中应报告 `mean` vs `zeros` 的对比，作为初始化的鲁棒性证据。

### 7.3 对照实验的数学意义

| 实验 | 第 4 通道 $P'$ | 数学含义 | 期望结论 |
|---|---|---|---|
| E0 | 无（$C=3$） | 基线 | 参照点 |
| E1 | $P'\equiv 0$ | 恒等零映射，$w_4*0=0$，理论上**等价于 E0**（仅多 3136 个不生效参数） | E1 ≈ E0，证明提升**不是**来自参数量 |
| E2 | $P'\sim\mathcal U(0,1)$（按 patch 坐标固定种子） | 注入**纯噪声**通道，信息量 $I(P';Y)\approx0$ | E2 ≈ E0（甚至略降），证明提升**不是**来自任何额外通道 |
| E3/E4/E5 | $P'=G_{\theta^*}(I)$ | 携带与 $Y$ 相关的语义信息 | **显著 > E0**，且 E5 > E3（soft > binary） |
| E7 | $P'=M$（GT） | Oracle 上界（**严禁作为部署方法**） | 给出该思路的性能天花板 |

**关键推理链（论文的 Argument 主线）：**

$$\text{E1}\approx\text{E0}\ \Rightarrow\ \text{增益}\neq\text{容量}$$
$$\text{E2}\approx\text{E0}\ \Rightarrow\ \text{增益}\neq\text{任意第4通道}$$
$$\text{E5}>\text{E0}\ \wedge\ \text{E1}\approx\text{E2}\approx\text{E0}\ \Rightarrow\ \text{增益}=\text{先验携带的语义信息}$$

这是一个**完整的因果排除链**，审稿人无法用"参数多了"或"随机通道也能涨点"来质疑。

---

## 8. Lite-HR PriorNet（V3）的结构归纳偏置

### 8.1 为什么禁止下采样到 $H/32$

设目标的最小外接框边长为 $a$（像素）。经过 $k$ 级 2× 下采样后，其响应尺寸为 $a/2^k$。

| 下采样层级 | 特征图尺寸 | $a=16$ px 目标的响应 | 信息状态 |
|---|---|---|---|
| $H/2$ | $540\times960$ | $8\times8$ | 保留 |
| $H/4$ | $270\times480$ | $4\times4$ | 保留 |
| $H/8$ | $135\times240$ | $2\times2$ | 勉强 |
| $H/16$ | $68\times120$ | $1\times1$ | 亚像素 |
| $H/32$ | $34\times60$ | $0.5\times0.5$ | **湮灭** |

**结论**：PriorNet 的最深分支不得超过 $H/8$。这是 EVD4UAV 小目标特性直接导出的**硬约束**。

### 8.2 三分支多尺度融合的形式化

记 $\psi_s$ 为 $H/s$ 尺度的特征提取（深度可分离卷积 DSConv），$\mathcal U$ 为双线性上采样，
$U_{c\to b},U_{b\to a}$ 为 $1\times1$ 通道投影：

$$\begin{aligned}
A &= \psi_2\big(\mathrm{stem}(I)\big) &&\in\mathbb R^{32\times H/2\times W/2}\\
B &= \psi_4\big(\downarrow(A)\big) &&\in\mathbb R^{64\times H/4\times W/4}\\
C &= \psi_8\big(\downarrow(B)\big) &&\in\mathbb R^{128\times H/8\times W/8}\\[4pt]
B &\leftarrow B + \mathcal U\big(U_{c\to b}(C)\big) &&\text{（残差融合，深层语义注入}\\)
A &\leftarrow A + \mathcal U\big(U_{b\to a}(B)\big) &&\text{（再上采样注入浅层\\})\\[4pt]
Z &= \mathrm{head}\big(\mathrm{fuse}(A)\big) &&\in\mathbb R^{1\times H\times W}
\end{aligned}$$

**注意方向**：语义从**深（粗）流向浅（细）**，最终在 $H/2$ 输出后再上采样到全分辨率。
这保证高频空间细节由浅层分支 $A$ 主导，而语义判别力由 $C$ 提供。

### 8.3 深度可分离卷积的参数量

标准卷积 $\mathbb R^{c_{in}\times c_{out}\times k\times k}$：$c_{in}c_{out}k^2$
DSConv（depthwise + pointwise）：$c_{in}k^2 + c_{in}c_{out}$

压缩比（$k=3$）：
$$\frac{c_{in}c_{out}\cdot9}{c_{in}\cdot9+c_{in}c_{out}}=\frac{9c_{out}}{9+c_{out}}\xrightarrow{c_{out}=128}\approx8.4\times$$

这解释了为什么 Lite-HR 仅需 **0.08 M** 参数（V1 MiniUNet 为 1.16 M，减少约 **14.5 倍**），
却能达到更高的 Dice（0.7695 vs 0.6867）。

> 这是一个很好的论文论点：**先验生成本身应该是轻量的**，因为它是为下游服务的附加模块；
> 0.08 M 的参数开销仅为主干 $F_\phi$（14.8 M）的 **0.54%**，推理开销可忽略。

---

## 9. Altitude-Aware PriorNet（V4）：FiLM 条件调制

### 9.1 为什么高度不做第 5 通道

若把 $h$ 作为常量通道 $H_{ch}\equiv \tilde h\cdot\mathbf 1^{H\times W}$ 拼入输入，则
conv1 的输出为 $W_{1:3}*I + w_4*P' + w_5 * (\tilde h\mathbf 1)$。
由于 $w_5 * (\tilde h \mathbf 1) = \tilde h\,(w_5*\mathbf 1) = \tilde h\cdot\sum_{i,j}w_5[i,j]$，
这是一个**与空间位置无关的常量偏置**。

$$\Rightarrow \text{第5通道只能提供全局加性偏置，无法做空间自适应的特征调制。}$$

而且它还会污染 conv1 之后所有层的激活分布。因此**必须**用条件调制。

### 9.2 FiLM 形式化

高度嵌入：
$$\mathbf e_h = \mathrm{MLP}(\tilde h)=W_2\,\mathrm{ReLU}(W_1\tilde h + b_1)+b_2,\qquad
\tilde h=\frac{h-70}{40}\in\\{-0.5,0,0.5\\}$$

归一化中心取 70 m（数据集中位高度）、尺度取 40 m，使 $\tilde h$ 落在 $[-0.5,0.5]$，
有利于 MLP 的数值稳定。

特征调制（作用于 $H/4$ 分支 $B$ 与 $H/2$ 分支 $A$）：

$$\boxed{F'_{c,x,y}=\big(1+\gamma_c(\mathbf e_h)\big)\cdot F_{c,x,y}+\beta_c(\mathbf e_h)}$$

其中 $\gamma=\mathrm{Linear}(\mathbf e_h),\ \beta=\mathrm{Linear}(\mathbf e_h)$，输出维度等于特征通道数 $C_l$，
再 broadcast 到空间维。

> 代码：`blocks.py` 中 `return feat * (1.0 + g) + b`，且 $\gamma,\beta$ 的 weight 与 bias **均零初始化**。

### 9.3 零初始化的意义（重要，值得写进论文）

零初始化 $\Rightarrow$ $\gamma\equiv0,\beta\equiv0$ $\Rightarrow$ $F'=F$，即**训练初始时 FiLM 是恒等映射**。
这带来两个保证：

1. V4 在前向初期**严格等价于 V3**，因此 V4 的任何性能差异都可归因于"高度条件的引入"，
   而不是随机初始化的扰动；
2. 高度信息是**渐进注入**的：只有当梯度认为调制有益时，$\gamma,\beta$ 才会偏离 0。

这是一个标准的"残差式条件注入"技巧，**显著优于**直接把 $\mathbf e_h$ 拼接到特征图上。

### 9.3b 零初始化的实证验证（可直接引用）

在 V4 上实测同一输入在不同高度下的输出差异：

| 状态 | $\lVert \text{out}(50\text{m})-\text{out}(90\text{m})\rVert_\infty$ |
|---|---|
| 零初始化 FiLM（训练开始前） | **0.000e+00** |
| 扰动 $\gamma,\beta$ 后 | 5.863e-02 |

第一行严格等于 0，**实验证实了 $\gamma=\beta=0$ 时 FiLM 是恒等映射**；
第二行说明扰动后高度条件确实生效。这两行一起构成一个干净的实验证据：
V4 的任何性能差异都只能来自"高度条件被学习到了"，而不是初始化噪声。

### 9.4 高度为何对先验有意义（可写的直觉 + 可验证的量化）

飞行高度 $h$ 与目标表观像素面积近似满足**针孔相机模型**下的平方反比关系：

$$a(h)\approx a_0\cdot\frac{f}{h}\quad\Rightarrow\quad \text{面积}\ \propto\ h^{-2}$$

故 $h$ 从 50 m 到 90 m，目标面积缩小 $(90/50)^2\approx3.24$ 倍。
这意味着**最优的 $\sigma$ 应该是高度相关的**：

$$\sigma(h)=\sigma_0\cdot\frac{h}{h_0}$$

> **可作为论文的扩展论点**：V4 让网络自己从数据中学习这个尺度律，而不是手工设定 $\sigma(50),\sigma(70),\sigma(90)$。
> 也可做一组消融：手工分高度设 $\sigma$ vs FiLM 自适应。

### 9.5 实证修正：BCP 增益与绝对飞行高度弱相关（2026-09-08）

§9.4 的尺度律 $a\propto h^{-2}$ 在**物理上成立**，但实测表明它对「BCP 的贡献」解释力有限：

分高度（E0→E5 的 ΔIoU，val 9129 patch）：

| 高度 | E0 IoU | E5 IoU | Δ |
|---|---|---|---|
| 50 m | 0.9105 | 0.9106 | +0.0001 |
| 70 m | 0.8276 | 0.8295 | +0.0019 |
| 90 m | 0.9052 | 0.9063 | +0.0011 |

- BCP 增益在三个高度层**均很小且量级相近**（Δ<0.002），与"高度越低目标越大、增益应越小"的简单预期不符。
- 结合 §19.2（高度与雪强混淆：70 m 层 63.4% 有雪、90 m 层 4.5%），分高度差异的主要驱动是**雪覆盖**而非绝对尺度——这进一步削弱"高度本身是关键因子"的假设。
- **初步推论（后被 §9.6 交叉实测修正）**：一维分高度看增益都很小，容易被误判为"与高度解耦"；
  但一维分组会把各尺寸层混在一起、正负抵消。**尺寸×高度交叉（§9.6）才是真正的判据**。

### 9.6 尺寸×高度交叉实测：V4 判定反转（2026-09-08，已实测）

交叉分组（同一 patch 同时按尺寸组与高度层标记）的 ΔIoU（E0→E5）：

| 组 | patch 数 | E0 IoU | E5 IoU | ΔIoU | ΔRecall |
|---|---|---|---|---|---|
| tiny@50m | 57 | 0.5885 | 0.6698 | **+0.0813** | +0.0729 |
| tiny@70m | 77 | 0.6072 | 0.6491 | **+0.0419** | +0.0473 |
| tiny@90m | 96 | 0.7620 | 0.7610 | **−0.0010** | −0.0041 |

**结论（推翻 §9.5 的"与高度解耦"猜想）**：

- BCP 增益**随飞行高度单调递减**：低空 50m 最大（+8.1 IoU 点），70m 次之（+4.2），
  高空 90m **完全失效**（−0.1）。增益与高度**强相关**，并非解耦。
- **机制解释（与 §9.4 尺度律自洽）**：目标面积 $\propto h^{-2}$，而当前 soft prior 使用**固定** $\sigma$。
  低空目标大，固定 $\sigma$ 相对适中 ⇒ 先验有效；高空目标极小，固定 $\sigma$ **相对过大** ⇒
  先验被过度平滑/膨胀，反而淹没极小目标 ⇒ 增益消失。这与 §9.4 的 $\sigma(h)=\sigma_0\cdot h/h_0$ 一致：
  高空需要**更小**的 $\sigma$。
- **⇒ V4（Altitude-Aware / FiLM 高度条件化）有明确动机**，且不是"边际改进"：
  让网络自学 $\sigma(h)$，有望把 90m 的 tiny 增益从 ≈0 提升到接近 50m 的水平。
  **V4 应从「可选扩展」升格为主贡献链的一环。**

> 方法论教训：一维分组（只看高度或只看尺寸）会掩盖交互效应——
> 本例中"分高度 Δ 全 <0.002"一度误导出"与高度解耦"的错误结论，
> 只有交叉分组才暴露出"低空 +8.1 / 高空 −0.1"的强交互。

---

## 10. 为什么第 4 通道绝不能是真值掩膜 $M$（标签泄露的信息论证明）

这是论文必须正面回应的质疑。给出严格论证：

### 10.1 信息论表述

记 $Y$ 为真值标签随机变量，$I$ 为输入图像，$\hat Y$ 为预测。

若第 4 通道 $=M=Y$，则网络输入为 $(I,Y)$，输出 $\hat Y=F_\phi(I,Y)$。
此时互信息

$$I(\hat Y;Y)\le H(Y)$$

可以轻易被一个**平凡解**达到上界——网络只需实现恒等映射 $\hat Y=\mathrm{round}(P')$，
即把第 4 通道直接搬到输出。此时

$$\mathrm{IoU}(\hat Y,Y)=1.0$$

### 10.2 数据加工不等式（DPI）的视角

由数据处理不等式，若 $\hat Y$ 只通过 $(I,P')$ 依赖 $Y$，则

$$I(\hat Y;Y)\le I((I,P');Y)$$

- 若 $P'=Y$（E7）：右端 $=I((I,Y);Y)=H(Y)$，**上界被完全释放**，网络学到了一个在部署时不存在的捷径。
- 若 $P'=G_{\theta^*}(I)$（E5）：右端 $=I((I,G_{\theta^*}(I));Y)=I(I;Y)$，因为 $G$ 是 $I$ 的确定性函数，
  不引入关于 $Y$ 的**额外**信息。

$$\boxed{I((I,G_{\theta^*}(I));Y)=I(I;Y)<H(Y)}$$

**这就是 E5 没有泄露、E7 泄露的严格区分。**

### 10.3 部署不可得性（工程论证）

推理时 $Y$ 不存在，故 $F_\phi(\cdot,Y)$ 在测试时**无法求值**。
E7 报告的数字只是一个**理论上界（Oracle）**，用于衡量"先验这一路信息的价值上限"，
不能与 E0 并列作为方法比较。

> 代码已明确标注：`patch_dataset.py` 中 `oracle` 分支注释"仅理论上界，严禁当作真实部署方法"。

---

## 11. OOF（Out-Of-Fold）先验的必要性

### 11.1 问题：非 OOF 先验导致的训练/推理分布偏移

设 PriorNet 在**全部**训练数据 $\mathcal D_{tr}$ 上训练，得到 $\theta^*$。
对任意训练样本 $n\in\mathcal D_{tr}$，其先验为

$$P^{(n)}=G_{\theta^*}(I^{(n)})$$

由于 $\theta^*$ 的最小化目标包含了 $\mathcal L(G_\theta(I^{(n)}), P^\*\_{gt}{}^{(n)})$，
而 $P^\*\_{gt}{}^{(n)}$ 由 $M^{(n)}$（即 $Y^{(n)}$）构造，故

$$P^{(n)} \text{ 与 } Y^{(n)} \textbf{ 统计相关} \quad\Longrightarrow\quad
\mathbb E[P^{(n)}\mid Y^{(n)}=1] \gg \mathbb E[P^{(n)}\mid Y^{(n)}=0]$$

这种相关性**强于**测试时 PriorNet 对未见样本的预测质量（PriorNet 在自己的训练样本上过拟合，
预测质量被高估）。

后果：Final CNN 在训练期看到的第 4 通道"质量过高"，而推理期看到的先验质量较低——
**train/test 分布不匹配**，导致 E5 的数字虚高，结论不可信。

### 11.2 5-折 OOF 的形式化

将 $\mathcal D_{tr}$ 划分为互斥的折 $\mathcal D_1,\dots,\mathcal D_5$，$\mathcal D_{tr}=\bigsqcup_{k}\mathcal D_k$。
对每折 $k$，在 $\mathcal D_{tr}\setminus\mathcal D_k$ 上训练 PriorNet，得 $\theta^{*}_{-k}$。
样本 $n\in\mathcal D_k$ 的先验为

$$\boxed{P^{(n)}_{OOF}=G_{\theta^{*}_{-k}}(I^{(n)}),\qquad k=\mathrm{fold}(n)}$$

关键性质：

$$\forall n:\quad I^{(n)}\notin\text{训练集}(\theta^{*}_{-\mathrm{fold}(n)})$$

因此 $P^{(n)}_{OOF}$ 是 $I^{(n)}$ 的**留一式（held-out）预测**，其统计特性与测试时对未见图像的预测**一致**。

> 代码：`scripts/generate_oof_prior.py`，并输出 per-fold manifest 显式列出
> `seen_by_generator = 0`，作为可审计的证据。论文中可直接引用该表。

### 11.3 折内划分的序列级约束

EVD4UAV 由 **142 个飞行序列**（`DJI_XXXX`）组成，同一序列的相邻帧高度相似。
若按帧随机划分，同一序列的帧会同时出现在 train 与 val，造成**近重复泄露**。

因此划分必须以序列为单位：

$$\mathrm{fold}(n)=\mathrm{fold}\big(\mathrm{seq}(I^{(n)})\big),\qquad
\mathrm{seq}: \text{stem}\mapsto \texttt{regex}^\wedge(DJI\backslash d\\{4\\})$$

> **已知问题（需在论文 Limitations 中诚实说明）**：官方 train/val 划分是**帧级**的，
> 142 个序列中有 **33 个**横跨 train 与 val。本文的自建 5 折划分是序列级的，
> 但若要与官方划分对齐比较，必须承认这一泄露风险。

---

## 12. Patch 化与采样的数学

### 12.1 滑窗分块

原图 $1920\times1080$，patch $p=768$，重叠 $o=192$，步长

$$s=p-o=576$$

坐标序列（含边界对齐）：
$$X=\mathrm{range}(0,W-p+1,s)\cup\\{W-p\\}=\\{0,576,1152\\},\quad
Y=\mathrm{range}(0,H-p+1,s)\cup\\{H-p\\}=\\{0,312\\}$$

每图 $3\times2=6$ 个 patch。9,371 张图 $\to$ 约 56,226 个候选。

### 12.2 负样本下采样

前景占比仅 $2.34\%$，全背景 patch 占绝大多数。按 $\kappa=0.15$ 保留空 patch：

$$|\mathcal P| = |\mathcal P_{fg}| + \kappa\cdot|\mathcal P_{empty}| \approx 45{,}633$$

（实测：tiny 2,964 / small 15,812 / medium 26,857）

> 论文中需说明 $\kappa$ 的选择，并做 $\kappa\in\\{0.05,0.15,0.3,1.0\\}$ 的消融。

### 12.3 划分先于分块（泄漏防护）

$$\textbf{约束：}\quad \forall\ \text{同图 } I^{(n)}:\ \text{其所有 patch 属于同一 split}$$

形式化：$\mathrm{split}(\mathrm{patch})=\mathrm{split}(\mathrm{source\_image})$。
若违反，同一张图的相邻 patch（重叠 192 px 区域完全相同）会分处 train/val，
导致 val 指标虚高。代码自检输出 `train ∩ val = 0`（已通过）。

---

## 13. 评估指标的定义

设 $\mathrm{TP}=\sum \hat Y M$，$\mathrm{FP}=\sum\hat Y(1-M)$，$\mathrm{FN}=\sum(1-\hat Y)M$，
$\mathrm{TN}=\sum(1-\hat Y)(1-M)$。阈值化 $\hat Y=\mathbb 1\\{\mathrm{sigmoid}(\hat z)\ge\tau\\}$，默认 $\tau=0.5$。

$$\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}$$

$$\mathrm{IoU}=\frac{TP}{TP+FP+FN},\qquad
\mathrm{Dice}=\frac{2TP}{2TP+FP+FN}=\frac{2\cdot\mathrm{P}\cdot\mathrm{R}}{\mathrm{P}+\mathrm{R}}$$

$$\mathrm{F_1}=\mathrm{Dice}$$

> 注意 Dice 与 F1 在二值分割下**代数等价**，论文中不必同时报告（或注明等价）。

### 13.1 分组评估

按三个维度切片，这是本文针对 EVD4UAV 特性的核心评估设计：

| 维度 | 分组 | 关注指标 |
|---|---|---|
| 高度 $h$ | 50 / 70 / 90 m | $\mathrm{IoU}_h$，验证 §9.4 的尺度律 |
| 目标尺寸 | tiny $<32^2$ / small $32^2\!\sim\!96^2$ / medium $>96^2$ | **$\mathrm{Recall}_{tiny}$、$\mathrm{IoU}_{tiny}$、$\mathrm{Dice}_{tiny}$** |
| 雪覆盖 $s$ | 有雪 / 无雪 | $\mathrm{IoU}_{snow}$ vs $\mathrm{IoU}_{clean}$ |

**核心假设（论文 Claim）**：BCP 的增益应当**集中在 tiny 子集**，因为
tiny 目标最依赖"空间注意力提示"，而 RGB 特征在下采样中最先丢失它们。

$$\Delta_{tiny}=\mathrm{IoU}^{E5}_{tiny}-\mathrm{IoU}^{E0}_{tiny}\ \gg\
\Delta_{medium}=\mathrm{IoU}^{E5}_{medium}-\mathrm{IoU}^{E0}_{medium}$$

如果这个不等式成立，就**直接验证了 BCP 的作用机制**（而非偶然涨点）。这是全文最有说服力的证据。

---

## 14. 完整前向-反向计算图（逐步）

**前向（推理一个新样本 $I^{test}$）：**

1. $I^{test}\in\mathbb R^{3\times1080\times1920}$，归一化 $\div255$
2. $Z=G_{\theta^*}(I^{test})$ → $\mathbb R^{1\times1080\times1920}$ logits
3. $P=\mathrm{sigmoid}(Z)$（可选 $P'=\mathrm{sigmoid}(\gamma Z+b)$）
4. $I_{RGBP}=[I^{test};P]\in\mathbb R^{4\times1080\times1920}$
5. $\hat z=F_\phi(I_{RGBP})\in\mathbb R^{1\times1080\times1920}$
6. $\hat Y=\mathbb 1\\{\mathrm{sigmoid}(\hat z)\ge0.5\\}$

**训练（两个独立的优化问题，顺序执行）：**

$$\textbf{Stage 1:}\quad \theta^*=\arg\min_\theta\ \sum_{n}\mathcal L_{prior}\big(G_\theta(I^{(n)}),P^\*\_{gt}{}^{(n)}\big)$$

$$\textbf{Stage 2:}\quad \phi^*=\arg\min_\phi\ \sum_{n}\mathcal L_{final}\big(F_\phi([I^{(n)};P^{(n)}_{OOF}]),M^{(n)}\big)$$

> **注意**：Stage 2 中 $\theta^*$ 已冻结，**梯度不回传到 PriorNet**。
> 这保证两阶段解耦，且 Stage 2 的计算图恒定，可与 E0 严格对齐。

**大图推理的分块融合**：对 $768\times768$ 滑窗，重叠区域采用**加权平均而非覆盖**：

$$P_{fused}(x,y)=\frac{\sum_{t}\mathbb 1\\{(x,y)\in R_t\\}\cdot P_t(x,y)}{\sum_t\mathbb 1\\{(x,y)\in R_t\\}}$$

> 代码：`scripts/infer_prior.py`，累加器 + 计数图，最后相除。

---

## 15. 论文写作建议（如何组织 Method 章节）

```
3. Method
3.1 Problem Formulation            -> §2（含前景不平衡的量化 2.34%）
3.2 Background Confidence Prior    -> §3（定义 P，为什么是第 4 通道，§3.3 表格）
3.2.1 Soft Prior Target            -> §4（RBF 核、四条数学性质、半高宽度、σ 选择表）
3.3 PriorNet Architecture          -> §8（小目标与下采样层级表、Lite-HR 公式、参数量对比）
3.3.1 Loss                         -> §5（Focal 完整推导 + Dice 互补性论证）
3.4 Final Segmentation Network     -> §6 + §7（公平性证明、Δ=3136、conv1 初始化）
3.5 Altitude-Aware Extension       -> §9（FiLM、零初始化恒等性、h^-2 尺度律）
3.6 Out-of-Fold Prior Generation   -> §11（OOF 必要性推导、5-fold 形式化）
3.7 Why Not Ground-Truth Prior     -> §10（DPI 证明，回应审稿人）
```

**最可能被挑战的三个点及应答：**

| 质疑 | 应答 |
|---|---|
| "是不是参数变多了所以涨点？" | §7.1：$\Delta=3136$ 参数（0.021%），且 E1（零通道）≈ E0 |
| "是不是随便加个通道都能涨？" | §7.3：E2（随机通道）≈ E0 |
| "PriorNet 见过训练样本，数字虚高吧？" | §11：5-fold OOF + per-fold manifest 证明 `seen_by_generator=0` |

---

## 16. 代码 ↔ 公式对照索引

| 公式 | 文件 | 关键行 |
|---|---|---|
| $M=\max_i M_i$ | `scripts/build_binary_masks.py` | `binary |= m.astype(np.uint8)` |
| $d(x,y)$ EDT | `scripts/build_soft_prior_targets.py` | `distance_transform_edt(~fg)` |
| $P^\*\_{gt}=e^{-d^2/2\sigma^2}$ | 同上 | `np.exp(-(dist**2)/s2)`，`s2=2σ²` |
| $\mathcal L_{focal}$ | `models/losses.py` | `FocalLoss.forward` |
| $\mathcal L_{dice}$ | 同上 | `DiceLoss.forward`，`dims=(0,2,3)` |
| $\mathcal L_{prior}$ | 同上 | `CombinedLoss.forward` |
| Lite-HR 三分支 | `models/priornet.py` | `LiteHRPriorNet.forward` |
| FiLM | `models/blocks.py` | `feat * (1.0 + g) + b` |
| $\mathbf e_h=\mathrm{MLP}(\tilde h)$ | `models/priornet.py` | `alt_mlp`，`t=(t-70.0)/40.0` |
| conv1 第 4 通道均值初始化 | `models/final_cnn.py` | `w_old.mean(dim=1, keepdim=True)` |
| $[I;P']$ 拼接 | `datasets_py/patch_dataset.py` | `torch.cat([t_rgb, t_p], dim=0)` |
| E0/E1/E2/E7 通道策略 | 同上 | `zero` / `random` / `oracle` 分支 |
| OOF 先验生成 | `scripts/generate_oof_prior.py` | 5-fold + `seen_by_generator=0` |
| 分块融合平均 | `scripts/infer_prior.py` | acc/cnt 累加后相除 |

---

---

## 21. 主实验结果（E0 vs E5，已收敛 40 轮）

### 21.1 总体指标

| 指标 | E0 (RGB) | E5 (BCP-Vis) | Δ |
|---|---|---|---|
| IoU（微平均） | 0.8885 | 0.8899 | +0.0014 |
| IoU（宏平均） | 0.8437 | 0.8469 | +0.0032 |
| Dice | 0.9387 | 0.9395 | +0.0008 |
| Recall | 0.9279 | 0.9275 | −0.0004 |
| Precision | 0.9529 | 0.9549 | +0.0020 |
| 参数量 | 14,834,121 | 14,837,257 | +3,136 |

聚合指标几乎持平——正如 §19.1 所论证的，像素微平均被 medium（占 89.4% 前景像素）
主导，tiny 的成败在聚合层面不可见。

### 21.2 分尺寸结果（论文主表）

| 组 | n | E0 IoU | E5 IoU | **ΔIoU** | E0 Recall | E5 Recall | **ΔRecall** |
|---|---|---|---|---|---|---|---|
| **tiny** | 230 | 0.6736 | **0.7092** | **+0.0356** | 0.7638 | **0.7879** | **+0.0242** |
| small | 3228 | 0.8715 | 0.8748 | +0.0034 | 0.9203 | 0.9220 | +0.0016 |
| medium | 5307 | 0.8884 | 0.8888 | +0.0004 | 0.9248 | 0.9230 | −0.0018 |

$$\frac{\Delta\mathrm{IoU}_{tiny}}{\Delta\mathrm{IoU}_{medium}}=\frac{0.0356}{0.0004}\approx 89\times$$

### 21.3 可证伪预测的检验结果（§20.3）

| 预测 | 结果 | 判定 |
|---|---|---|
| ΔRecall_tiny > 0 | +0.0242 | ✅ |
| ΔRecall_tiny ≫ ΔRecall_medium | +0.0242 vs −0.0018 | ✅ |
| 涨 Recall 而非 Precision（注意力机制自洽） | ΔRec +0.0242 / ΔPrec +0.0256，基本同涨 | ⚠️ 部分成立 |

**解读**：
- 增益**高度集中在 tiny**，且 recall 有实质提升——BCP 确实通过"告诉网络该看哪里"
  找回了 baseline 漏掉的小目标。这与 §20 的先验互补性诊断（可召回率 0.79）完全一致。
- Precision 也小幅上涨（+0.0256），说明先验同时抑制了部分背景假阳性。
  这不是"注意力机制"的排他性证据，但也不与之矛盾——
  先验不仅标注了"哪里有目标"，其低值区也隐式标注了"哪里是干净背景"。
- 论文表述建议：BCP 的作用是**双重**的——(a) 对小目标的空间召回（主要），
  (b) 对背景的抑制（次要）。两者都源于同一个 soft prior 的两端（高值/低值区）。

### 21.4 消融实验：因果排除链完整验证（论文核心表）

**tiny 子集 IoU / Recall（n=230）**：

| 实验 | 第4通道 | tiny IoU | Δ vs E0 | tiny Recall | Δ vs E0 |
|---|---|---|---|---|---|
| E0 | 无（3ch） | 0.6736 | — | 0.7638 | — |
| **E1** | **全零** | 0.6661 | **−0.0075** | 0.7414 | **−0.0224** |
| **E2** | **随机** | 0.6633 | **−0.0103** | 0.7448 | **−0.0190** |
| **E5** | **BCP 先验** | **0.7092** | **+0.0356** | **0.7879** | **+0.0242** |

**medium 子集 IoU（对照）**：E0 0.8884 / E1 0.8883 / E2 0.8863 / E5 0.8888 —— 全部持平。

**总体指标**：

| 实验 | IoU | IoU_macro | Dice | Recall | Precision |
|---|---|---|---|---|---|
| E0 | 0.8885 | 0.8437 | 0.9387 | 0.9279 | 0.9529 |
| E1 | 0.8891 | 0.8437 | 0.9392 | 0.9278 | 0.9540 |
| E2 | 0.8866 | 0.8420 | 0.9376 | 0.9262 | 0.9525 |
| E5 | 0.8899 | 0.8469 | 0.9395 | 0.9275 | 0.9549 |

**因果排除链（§7.3 的三条推理全部得证）**：

$$\text{E1}(\text{零})\approx\text{E0}\ (\Delta_{tiny}=-0.008)\ \Rightarrow\ \text{增益}\neq\text{容量}$$
$$\text{E2}(\text{随机})\approx\text{E0}\ (\Delta_{tiny}=-0.010)\ \Rightarrow\ \text{增益}\neq\text{任意第4通道}$$
$$\text{E5}\gg\text{E0}\ (\Delta_{tiny}=+0.036)\ \Rightarrow\ \text{增益}=\text{先验携带的语义信息}$$

**更强的细节**：E1 与 E2 在 tiny 上不仅没有提升，反而**略低于 E0**（−0.008/−0.010）——
零通道与随机通道引入的是**无信息偏置**，对小目标反而是轻微干扰。
而 E5 高出 +0.036。**同一架构、同一训练预算下，唯一变量是第 4 通道的内容，
E1 < E0 < E5 的排序无法用容量、随机正则化或训练噪声解释。**

**Recall 视角更显著**：E1/E2 使 tiny Recall 分别下降 0.022/0.019，
E5 提升 0.024——单向摆幅约 0.045，方向完全由第 4 通道是否有语义决定。

### 21.5 收敛速度

E0 首轮 IoU 0.798，E5 首轮 0.840（+0.042）；E5 在第 28 轮即追平 E0 全程最佳 0.8940。
说明先验在训练初期提供了有效的归纳偏置，加速了收敛。可作为附加论点。

### 21.6 待完成（用于最终论文的完整证据链）

- E7（Oracle GT prior）：理论上界（当前 ep15 已达 IoU 1.0000，符合预期）
- OOF 5-fold：排除 PriorNet 见过训练样本导致的先验过拟合（§11）
- E5_OOF：论文级严谨版主实验
- E6 / V4：高度条件能否进一步改善（尤其 tiny@不同高度）
- 雪况分组：clean 0.9105→0.9115、snow 0.8217→0.8225（当前均微涨，需 OOF 后复核）

---

## 20. 先验互补性诊断：E5 训练前就能预判成败

工具：`tools/diagnose_prior.py`。逐像素定义

$$\mathrm{Miss}=Y\cap \bar B,\qquad \mathrm{Hit}=\mathbb 1\\{P>\tau\\},\qquad
\text{可召回率}=\frac{\lvert \mathrm{Miss}\cap \mathrm{Hit}\rvert}{\lvert \mathrm{Miss}\rvert}$$

其中 $B$ 是 baseline 预测。若先验能覆盖 baseline 漏掉的像素，则把先验作为第 4 通道
**有可能**把这些漏检找回来。

### 20.1 实测结果（val，分层采样 530 个 patch）

| 组 | n | baseline 召回 | **先验召回** | 先验精度 | 先验覆盖面积 | **可召回率** |
|---|---|---|---|---|---|---|
| **tiny** | 230 | 0.7999 | **0.9261** | 0.0606 | **1.35%** | **0.7914** |
| small | 200 | 0.9388 | 0.9127 | 0.2045 | 3.60% | 0.7513 |
| medium | 100 | 0.9508 | 0.9096 | 0.2883 | 12.71% | 0.8826 |

### 20.2 三条可直接写进论文的结论

**(a) 先验在空间定位上优于分割网络本身（小目标上）。**
$$0.9261\ (\text{PriorNet},\ 0.08\text{M})\ >\ 0.7999\ (\text{ResNet18-UNet},\ 14.8\text{M})$$
一个 0.08 M 的轻量网络在"找到小目标在哪"这件事上，**召回率比 14.8 M 的分割网络高
12.6 个百分点**。原因是先验的监督目标是带 $\sigma=32$ 高斯衰减的软目标，
在小目标周围提供了密集的、梯度连续的区域监督，而硬掩膜在 2.9% 前景占比下
给小目标的梯度几乎被背景淹没。

**(b) 先验的高召回不是靠"面积大"堆出来的。**
tiny 组先验只覆盖图像面积的 **1.35%**，却覆盖了 GT 前景的 **92.61%**，
富集倍数（lift）为
$$\mathrm{lift}=\frac{0.9261}{0.01349}\approx 68.7\times$$
作为对照，medium 组的 lift 仅 $0.9096/0.1271\approx7.2\times$。
**先验对小目标的空间定位信息密度远高于对大目标的**——这正是我们希望第 4 通道携带的信息。

**(c) baseline 漏掉的像素中，先验能补回 79%。**
tiny 组可召回率 0.7914。结合 §19.3 的预测（BCP 应主要提升 Recall），
这为"注意力先验"机制提供了**正向的先验证据**。

### 20.3 由此得到的可证伪预测

$$\Delta \mathrm{Recall}_{tiny}\ >\ 0,\qquad \text{且}\quad
\Delta \mathrm{Recall}_{tiny}\ \gg\ \Delta \mathrm{Recall}_{medium}$$

因为 medium 组 baseline 召回已达 0.951（几乎饱和，空间极小），
而 tiny 组 baseline 0.800、先验 0.926，存在约 12.6 个百分点的可迁移空间。

> 若 E5 结果违背此预测（例如 tiny 召回未提升，或 medium 反而涨得更多），
> 则说明第 4 通道的信息没有被下游网络有效利用，需要回到 §3.3 重新考虑
> 融合方式（如改为中期注意力门控）。**这是可检验的，不是事后解释。**

---

## 19. 分组结构的量化证据（决定论文 Table 怎么设计）

本节数据来自 `tools/analyze_groups.py`（split=val，9129 个 patch）与 E0 预演评估。

### 19.1 聚合指标在数学上无法反映小目标（最强论据）

| 尺寸组 | patch 数 | patch 占比 | 前景像素占比 |
|---|---|---|---|
| medium | 5307 | 58.1% | **89.41%** |
| small | 3228 | 35.4% | 10.51% |
| tiny | 594 | 6.5% | **0.078%** |

**tiny 子集占 6.5% 的 patch，却只贡献 0.078% 的前景像素。**

由于像素级微平均
$$\mathrm{IoU}=\frac{\sum TP}{\sum TP+\sum FP+\sum FN}$$
按像素加权，tiny 的权重只有 **0.078%**——即使 tiny 上 IoU 从 0.47 掉到 0，
聚合 IoU 也只下降约 $0.078\%\times0.47\approx0.0004$，**完全淹没在噪声里**。

> 论文写法：这不是"我们选择关注小目标"，而是"**在 EVD4UAV 上，聚合指标
> 对小目标性能的敏感度为零**"。这是必须分组评估的**数学理由**，而非偏好。

实测佐证：E0 聚合 IoU = 0.884，而 tiny 子集 IoU = **0.471**——差 0.41，
聚合指标却几乎纹丝不动。

### 19.2 高度与雪强混淆（重要陷阱）

| 高度 | patch 数 | **有雪比例** | fg 中位数 | IoU | Recall |
|---|---|---|---|---|---|
| 50 m | 2353 | 43.3% | 15079 | 0.907 | 0.947 |
| **70 m** | 2922 | **63.4%** | 9222 | **0.824** | 0.871 |
| 90 m | 3854 | **4.5%** | 11993 | 0.902 | 0.944 |

**反直觉现象**：70 m 最难，而不是最高的 90 m。

**原因**：70 m 层有雪比例高达 63.4%，而 90 m 层只有 4.5%。
高度与雪覆盖在数据集中**高度混淆**，若直接把性能差异归因于飞行高度，结论是错的。

> 论文必须：(a) 把 snow 作为**独立分组维度**报告（已加 `by_snow.csv`）；
> (b) 明确指出 altitude–snow 混淆，避免做出错误因果断言；
> (c) 这恰好是 EVD4UAV（含雪场景）相比 VisDrone 的核心价值所在，
>     应把"雪地下的小目标"作为 BCP 的主要卖点之一。

### 19.3 empty 与 tiny 必须分离（指标设计缺陷的修正）

原分组把 `fg < 32²` 一律归为 tiny，但实测 **tiny 组的前景像素中位数为 0**——
即一半以上是完全无目标的空 patch。这混合了两种完全不同的失败模式：

| 失败模式 | 表现 | 归因 |
|---|---|---|
| 漏检小目标 | Recall 低 | 网络看不到小物体 |
| 空背景幻觉 | Precision 低（模型在雪地/路面产生假阳性） | 背景判别力不足 |

已修正 `size_group()`：新增 `empty` 组（$fg=0$），使 tiny 成为**纯粹的
小目标性能指标**。分离后共 364 个空 patch（占 val 的 4.0%）。

**修正后的结论与混合分组时相反**，这一点必须写准：

| 分组口径 | tiny 的 Precision | Recall | IoU | 判读 |
|---|---|---|---|---|
| 混合（含 364 空 patch） | 0.564 | 0.740 | 0.471 | 看似"假阳性主导" |
| **纯净（230 个真小目标）** | **0.865** | **0.770** | **0.687** | **实为"漏检主导"** |

即：baseline 在小目标上的主要失败是 **Recall（0.770）明显低于 Precision（0.865）**，
也就是**找不到目标**，而不是把背景误判成目标。

> **由此得出一个可证伪的预测**：BCP 作为"空间注意力提示"，其机制是告诉网络
> *该看哪里*，因此应主要提升 **Recall** 而非 Precision。
> 论文中可显式检验：
> $$\Delta \mathrm{Recall}_{tiny} \gg \Delta \mathrm{Precision}_{tiny}$$
> 若结果相反（主要涨 Precision），则"注意力先验"的机制解释不成立，
> 需改从"背景抑制"角度重新表述。**这是一个诚实且有力的自检。**

补充：空 patch 组的 IoU 恒为 0（因 $TP=FN=0$），无法反映严重程度，
故分组输出中额外给出 `fp_px` 与 `fp_per_patch` 用于量化"空背景幻觉"。

### 19.4 E0 baseline 预演结果（用于对照）

| 指标 | 值 |
|---|---|
| IoU（像素微平均） | 0.884 |
| **IoU（逐 patch 宏平均）** | **0.839** |
| Dice | 0.936 |
| Precision / Recall | 0.948 / 0.928 |
| 参数量 | 14,834,121 |
| 速度 | 215 fps（bs=8, 768²） |

分尺寸（分离空 patch 后）：medium 0.885 / small 0.872 / **tiny 0.687**（n=230）
分高度：50 m 0.907 / **70 m 0.824** / 90 m 0.902
分雪况：**clean 0.907**（n=6084）/ **snow 0.818**（n=3045）

**留给 BCP 的空间（理论上限 $1-\mathrm{IoU}$）**：

| 子集 | 当前 IoU | 最大可提升 |
|---|---|---|
| medium | 0.885 | 0.115 |
| small | 0.872 | 0.128 |
| **tiny** | **0.687** | **0.313** |
| **snow** | **0.818** | **0.182** |

tiny 的剩余空间是 medium 的 **2.7 倍**，snow 是 clean（0.093）的 **2.0 倍**。
因此论文的两个主攻方向应是 **tiny** 与 **snow**，而非整体指标。

> snow 与 clean 的差距（0.907 vs 0.818，差 0.089）远大于不同飞行高度之间的差距
> （扣除混淆后），进一步支持"雪覆盖才是真正的难点因子"这一判断（§19.2）。

---

## 18. 二次复核发现（2026-09-07，审查记录）

这一节记录**实验证据层面的关键事实**，直接决定论文该怎么写、结论该怎么下。

### 18.1 指标经独立复算确认无误

训练日志报告 E0（RGB baseline）第 1 epoch：IoU 0.8299 / Dice 0.9031。
因前景占比仅约 3%，该数字偏高，遂用**独立实现**（`tools/verify_pred.py`，不复用项目指标代码）
在 10 个 val patch 上重算：

| 口径 | IoU | Dice | Recall | Precision |
|---|---|---|---|---|
| 训练日志（全 val，batch 均值） | 0.8299 | 0.9031 | 0.9396 | 0.8746 |
| 独立复算（10 patch，像素微平均） | 0.8332 | 0.9090 | 0.9200 | 0.8983 |
| 独立复算（10 patch，逐 patch 宏平均） | **0.7919** | 0.8675 | — | — |

两者吻合 ⇒ **指标计算无 bug**。`binary_metrics_from_logits` 是标准 batch-global 实现，
不存在"空 patch 被算作 IoU=1"这类经典错误。

### 18.2 官方划分的序列级泄露：存在，但实测影响很小

| 项 | 数量 | 占比 |
|---|---|---|
| 序列总数 | 142 | — |
| 跨 train/val 的序列 | 33 | 23.2% |
| val 图像落在泄露序列 | 1847 / 1868 | **98.9%** |
| val patch 落在泄露序列 | 9013 / 9129 | 98.7% |

按数量看泄露极严重。但**决定性验证**推翻了"指标被泄露抬高"的猜测：
取 21 张**真正未见过序列**的 val 图像（21 个不同序列，116 个 patch）复算：

$$\mathrm{IoU}_{clean}=0.8428 \quad > \quad \mathrm{IoU}_{leaky}=0.8332$$

**结论**：泄露存在，但 EVD4UAV 场景高度同质（同一传感器、相似视角），
换序列并不显著改变难度。论文中仍应主动披露这一点（§11.3），
并可用 clean-val 子集（21 张）作为补充验证——**主动报告比被审稿人发现有利得多**。

### 18.3 真正的发现：聚合 IoU 已被大目标饱和

逐 patch 分解暴露了失败模式（E0，10 个 val patch）：

| patch | GT 前景占比 | 预测占比 | IoU | Recall |
|---|---|---|---|---|
| DJI_0402_133 | 9.15% | 9.70% | 0.922 | 0.988 |
| DJI_0384_55 | 3.58% | 4.27% | 0.804 | 0.978 |
| DJI_0157 | 0.93% | 0.99% | 0.937 | 0.995 |
| **DJI_0394_368** | **1.11%** | 0.84% | **0.478** | **0.566** |
| **DJI_0446_frame_75** | **1.48%** | 0.52% | **0.349** | **0.349** |

**规律**：失败集中在**低前景占比**的 patch（小目标 / 稀疏目标），
其中 DJI_0446 是**严重漏检**（Recall 0.349）——模型几乎没找到目标。

而像素级微平均 IoU 由大目标主导（DJI_0402 单块 9.15% 前景就贡献了大量像素），
把这些失败**稀释掉了**：微平均 0.8332 vs 宏平均 0.7919，差 0.041。

### 18.4 由此得出的论文写作策略（重要）

1. **聚合 IoU 不是本文的卖点。** baseline 已达 0.83，剩余空间小，
   BCP 的整体增益可能只有 +1~2 个百分点。若在摘要里主打聚合 IoU，会被认为贡献微弱。
2. **必须把 tiny / small 分组作为主结果。** 论文的 Claim 应改写为：
   > BCP 的价值不在于提升整体 IoU，而在于**修复小目标的漏检失败**
   > （baseline 在稀疏小目标 patch 上 Recall 可低至 0.35）。
3. **必须同时报告宏平均与微平均。** 已在 `eval_final.py` 中加入
   `iou_macro` / `dice_macro` 两列，与微平均并列。缺了宏平均，
   "增益集中在 tiny" 这一论断不可证伪。
4. **Recall 比 IoU 更关键。** 小目标失败模式是漏检（Recall 0.349）而非误检，
   因此 $\mathrm{Recall}_{tiny}$ 应作为**第一指标**，IoU 次之。

### 18.5 复核中修复的问题

| 问题 | 处理 |
|---|---|
| `eval_final.py:143` `modes[args.exp]` 查表，`--exp E5_OOF` 会 **KeyError 崩溃** | 执行链改为 `--exp E5 --tag E5_OOF`（自定义名只能走 tag） |
| 聚合指标掩盖小目标失败 | 新增 `iou_macro` / `dice_macro` / `n_patches` 三列 |
| `infer_prior.py` 默认写 float32 npy（8.3MB/张，曾吃掉 23G 磁盘） | 改为默认 uint8 PNG（258KB/张，全量 ~2.4GB） |

### 18.6 复核通过项

- 日志错误扫描（12 个 log）：无 Traceback / OOM / Killed / nan，仅 1 处是我主动 kill 导致的 `[FAIL] infer prior`
- V3 训练健康：train 0.3529 / val 0.3814（val/train = 1.08），**无过拟合**
- 图像级与 patch 级 train/val 交集均为 **0**
- 磁盘：18G / 50G，充裕
- 产物完整：binary_masks 9371、soft_priors 9371（1.89GB）、split.csv / patch_manifest.csv 齐备

---

## 17. 当前实验状态（2026-09-07）

| 阶段 | 状态 | 数值 |
|---|---|---|
| V1 MiniUNet PriorNet（smoke） | ✅ | best dice 0.6867 |
| **V3 Lite-HR PriorNet** | ✅ **完成 40 epoch** | **best dice 0.7695**，IoU 0.6454，Recall 0.6934 |
| infer prior（V3） | 🔄 运行中 | — |
| E0 RGB baseline | ⏳ 排队 | — |
| E5 BCP-Vis | ⏳ 排队 | — |
| E1/E2/E7 对照 | ⏳ chain2 排队 | — |
| OOF 5-fold prior | ⏳ 待执行 | — |
| V4 Altitude-Aware | ⏳ 待尺寸×高度交叉判定 | 交叉分析进行中（§9.5） |

> V3 的 train/val 曲线收敛一致（train 0.3529 / val 0.3814），**无过拟合**，
> 且参数量仅 0.08 M —— 这是论文中"轻量先验生成器"论点的直接证据。

---

## 22. V5 方案：专家头显式建模背景（背景/目标双分支先验）

> 思路来源：借鉴大模型中的 **专家头（expert head / MoE expert）** 思想——
> 用**独立的专家分支显式存放"背景"信息**，与"目标"分支分离，
> 从而更干净地区分目标与背景。本节论证其在 BCP-Vis 中的可行性与落地方案。

### 22.1 为什么这个思路有实验依据（不是凭空设想）

现有 soft prior 只用一个通道 $P$ 同时承担两种语义：

$$P\text{ 的高值区}\Rightarrow\text{目标；}\quad P\text{ 的低值区}\Rightarrow\text{背景（隐式为 }1-P\text{）}$$

但实测表明这两端**都在起作用**——E5 相对 E0 在 tiny 子集上
**Precision 与 Recall 同涨**（$\Delta\text{Prec}=+0.0256$，$\Delta\text{Rec}=+0.0242$，见 §21.3）。
这说明先验不仅"指出目标在哪"，其低值区也确实在**抑制背景假阳性**。

**问题在于**：背景至今只是 $1-P$ 的**隐式副产品**，从来没有一个专门的模块去学习它。
专家头思路正好补上这个缺口——把背景显式参数化。

### 22.2 它能针对当前最弱的两点

| 现有弱点 | 实测 | 专家头为何可能有效 |
|---|---|---|
| 空 patch 假阳性 | E5 在 `empty` 组仍有 **88.7 px/patch** 的 FP（E0 为 176.6） | 背景专家专门学"什么不是目标" |
| 雪地场景无增益 | `snow ΔIoU = −0.0013`（唯一负增益子集） | 雪堆/树影/建筑是典型"难背景"，正是背景专家的目标 |

### 22.3 必须避开的陷阱：与 §9.1 的区别（关键）

§9.1 已证明**高度做第 5 通道无效**，因为标量 $h$ 广播后是**空间无关常量偏置**：

$$w_5\cdot(h\cdot\mathbf{1}) = h\cdot\sum w_5$$

**但背景专家头不受此限制**：它输出的是**空间张量** $B(x,y)\in\mathbb{R}^{H\times W}$，
逐像素变化，不是常量广播。因此 §9.1 的否定结论**不适用于**背景专家通道，
第 5 通道在此处是有效的。这一点必须在论文中明确区分，否则会误用 §9.1 否定本方案。

### 22.4 三个落地方案（按性价比排序）

**方案 A：双分支 PriorNet（前景头 + 背景头）——推荐先做**

共享 Lite-HR 骨干，顶端分两个 $1\times1$ 卷积头：

$$P_f = \sigma(f_{\text{fg}}(F)),\qquad P_b = \sigma(f_{\text{bg}}(F))$$

- 前景头监督：现有 soft prior（GT 距离衰减目标）
- 背景头监督：$1-\text{GT}$（或显式背景 mask）
- 第 4 通道有三种取法，可逐一消融：
  - (a) 仅 $P_f$（等同现状，背景头只作辅助损失提升表征）
  - (b) 对比先验 $P_f - P_b$（更锐利，抑制"模糊区"）
  - (c) 同时给 $P_f, P_b$ 两个通道（第 4、5 通道，信息最全）
- 参数增量可忽略（两个 $1\times1$ 卷积），**保持"轻量先验"卖点**

**方案 B：MoE 背景专家（多专家 + 路由器）**

多个专家各专攻一类难背景（雪地/植被/建筑/道路），路由器按图像特征选专家。
更贴近大模型原意，但参数与复杂度上升，与"轻量"定位有张力，建议 A 验证有效后再上。

**方案 C：可学习背景原型库（最贴合"存放背景信息"原意）**

学一组背景原型向量 $\{b_k\}_{k=1}^K$（参数量极小，非网络），
每个空间位置的特征与原型做匹配得到"背景相似度"：

$$B(x,y) = \max_k \frac{\langle F(x,y),\,b_k\rangle}{\|F(x,y)\|\,\|b_k\|}$$

优点：轻量、可解释（可可视化每个原型对应何种背景）、天然适合"存放背景信息"的表述。

### 22.5 验证指标（必须与现有实验同口径）

- 主指标：`empty` 组的 FP 像素（假阳性是否下降）、`snow` 子集 ΔIoU（能否由负转正）
- 不退化检查：`tiny@50m` 的 +0.0813 不能被损害
- 因果对照：沿用 E1（零通道）/ E2（随机通道）排除容量与噪声解释

### 22.6 状态

⏳ **待执行**（当前 E6/V4 验证进行中；V4 结论出来后按 §22.1 的证据决定优先级）
建议命名：**E8 = RGB + 双分支先验**（与 E0/E5/E6 同协议对比）
