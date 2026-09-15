# BCP-Vis 相关工作（相似论文清单）

> 用途：论文 Related Work / 对比讨论素材。按三个创新点分组。
> 检索日期：2026-09-13（WebSearch）。
> 说明：★= 概念最贴合；○= 相关但机制不同；⚠= 对应已失效的创新点③（UAP 重评未通过，见 PAPER_MATERIALS）。

---

## 创新点①：BCP 先验通道 + 因果验证（learned foreground/background prior as explicit guidance）

### ★ Niu et al. 2024 — "A new framework for improving semantic segmentation in aerial imagery"
- 出处：*Frontiers in Remote Sensing*, 2024, DOI 10.3389/frsen.2024.1370697
- 核心：提出 **Foreground Precedence Estimation (FPE)**——用 1×1 卷积从 FPN 特征生成前景掩码
  `M_fg ∈ R^{1×H×W}`，作为显式先验与原始特征聚合（`B = δ(M_fg · I)`），同时建模前景/背景上下文；
  配套 Small Object Edge Alignment (SOEA) 与 foreground-saliency guided loss。
- 贴合度：最高。它"学习一张前景置信先验图并注入下游分割"的思想与 BCP 几乎一致。
- 与我们的差异：
  1. 它把先验用作**网络内部特征加权**（concat/聚合），我们是作为**第4输入通道**给下游 CNN，形式更直接、可解释。
  2. **没有做零/随机先验消融**来证伪"容量/正则化"解释——这正是 BCP 因果验证的独创处。
  3. 先验是二值的"前景掩码"，我们是 soft 连续置信（distance-transform + Gaussian decay）。

### ○ "Improving Semantic Segmentation in Aerial Imagery via Graph Reasoning and Disentangled Learning" (OpenReview)
- 同样用 FPE（foreground prior estimation）+ boundary alignment，是 Niu 思想的同族延伸。

### ○ SD-Grounded-SAM / 显式 saliency-depth 先验
- 用外部先验图（显著性、深度）作为分割引导信号。机制类似"先验即引导"，但是**外部模型生成的先验**，
  不是端到端学习的；与 BCP（PriorNet 端到端预测）不同。

### ○ HOSU（DINOv2/CLIP 基础模型先验）
- 借基础模型的语义先验增强小目标。属于"用强先验补小目标"，但先验来源不同。

---

## 创新点②：双维度专家级联（高度/尺度专家 + 背景/语义专家，串联）

### ★ UESDNet 2025 — "UAV Visual Multitask Perception Via Expert-Guided Prior With Mutual Optimization of Segmentation and Detection"
- 出处：*IEEE JSTARS*, 2025, DOI 10.1109/jstars.2025.3643918
- 核心：构建 STUDataset（含 UAV **ego-information：姿态/高度**），提出
  **ExpertSync 模块**把"object-region 空间一致性先验"用 UAV 自信息转化为可学习优化机制，
  缓解遮挡导致的漏检/分割破碎。命名上就是"expert-guided prior"。
- 贴合度：很高。它把**高度/姿态等 UAV 状态先验**引入分割，且用"expert"措辞，与我们的
  第1级高度专家思想相近。
- 与我们的差异：它是 seg+det 多任务互优化框架，我们用高度做 FiLM/MoE 条件调制第4通道先验；
  它也没有"先高度后背景"的级联两正交维度设计。

### ★ SIFDAL 2024 — "Scale-Invariant Feature Disentanglement via Adversarial Learning for UAV-based Object Detection"
- 出处：arXiv 2405.15465（github: 1e12Leon/SIFDAL），构建 State-Air 数据集（含 IMU 高度）。
- 核心：SIFD 模块把特征通道拆成 **scale-related / scale-invariant** 两组，scale-related 用
  **IMU 高度标签**监督学习，推理时丢弃 scale-related 仅留不变特征；
  AFL 用对抗训练强化解耦。
- 贴合度：很高。它**显式用高度监督来分离尺度因子**——与"目标面积 ∝ h⁻²、最优先验 σ 应随高度变"
  的动机完全一致，是创新点②中高度维度的强佐证。
- 与我们的差异：它"丢弃"尺度特征，我们"用高度条件调制"先验（保留并校正）；目标不同（检测 vs 分割）。

### ○ HASS — Height-Aware Feature-Scale Adaptive RT-DETR
- 高度感知的特征尺度自适应检测头。同族"高度→尺度"思路。

### ○ AltiDet — Altitude-informed fusion pyramid
- 用高度信息指导特征金字塔融合。同族。

---

## 创新点②补充：专家头 / 多专家级联组合（user 2026-09-13 追问"专家头结合"）
> 对应创新点②的第2级"背景专家头"（E8: P_f ⊙ (1−P_b)）与第1级"尺度/高度专家"的组合机制。
> 结论性发现：**文献中"多专家结合"几乎都用 加法 / 注意力 / 门控 融合，而非相乘抑制**，
> 与我们对 V5-A 失败的诊断（相乘 P_f⊙(1−P_b) 退化为 P_f²，压掉 tiny 软光晕）一致——
> 若重做第2级，应改相乘为"背景修正分支 + 注意力门控"（参照 DEDBNet / SA-MoE）。

### ★ DEDBNet 2024/2025 — "DoG-enhanced dual-branch object detection network"
- 出处：*Signal Processing* (DSP) 2025, DOI 10.1016/j.dsp.2024.104789（亦 ScienceDirect 同文）
- 核心：**双分支**——base 分支负责检测目标，DWB(weaken-background) 分支用**特征级注意力修正**
  检测结果；融合用 **SMCD（Self-Mutual-Correcter）+ MCA（Map Channel Attention）**。
- 贴合度：最高（对应第2级"背景专家头"）。它是用"独立背景修正分支 + 注意力"而非相乘，
  这正是我们该借鉴的温和融合形式。

### ★ HRDBNet 2025 — "High Resolution Dual Branch Network for Complex Scenario Small Object Detection on Drone View"
- 出处：*Aerospace Frontiers Conf.* (Springer), 2025
- 核心：**双分支分别检测 object 与 background**，用 contrastive loss 拉大 object/background 特征区分度实现背景抑制；
  在 CS-Drone 达 43.4% AP。
- 贴合度：高。object/background 解耦的双分支思想与"背景专家头"同源。

### ★ SA-MoE — "Scale-Aware Mixture of Experts" (fine-grained vegetation segmentation)
- 核心：**空间门控网络**（温度极化 τ=0.5 锐化专家权重差）+ **异构专家组 5 并行分支**
  （pixel expert, 3 个不同 dilation 的 spatial expert, global avg-pool expert），**动态 pixel-level 加权融合**；
  **extra-small (XS) 目标 recall +3.21pp**。
- 贴合度：高（对应第1级尺度专家）。它不仅做尺度路由，还明确报告 tiny 目标 recall 提升——
  与"增益集中于 tiny"的论点互相印证，是创新点②第1级的有力佐证。

### ★ DINOv3-LoRA with Mixture-of-Experts Decoding for UAV RS Seg (IEEE 2025)
- 核心：**CBAM-guided MoE decoder**，路由门控把特征动态分派给多个 iterative experts + shared expert，
  让大均匀区 / 边界敏感区 / **小前景目标**分别被精细化。
- 贴合度：高。多专家 decoder 处理"小前景目标"异质区域，命名即 MoE，与专家头思想一致。

### ○ Conv-LoRA+MoE（SAM for RS，ICLR-2024 清华团队思路）
- 8 个并行卷积专家处理 1/2x~8x 不同尺度，轻量 MLP 门控动态选专家；仅激活 1–2 个，计算降 67%。
- 尺度的专家路由思路（对应第1级）。

### ○ Mixture-of-experts for semantic segmentation of remote sensing (SPIE ICIPAI 2024, He et al.)
- Swin Transformer + MoE，不同尺度/场景激活不同专家；decoder 加 channel attention。

### ○ MFIAE (ICASSP 2025) — adaptive disturbance sparse MoE for panoptic RS
- 自适应扰动稀疏 MoE，提升小目标与泛化。

### ○ MEKD-UAVSeg — conflict-suppressed heterogeneous expert distillation
- 训练期用 Transformer 专家 + Mamba 专家（仅训练期），推理保持紧凑 CNN；**UAV-aware density / hard-region priors** 强调小目标密集区。
- 思路："专家仅训练期用、推理轻量"——与"先验作引导、不增加推理负担"精神相通。

---

## 创新点③：不确定性感知先验融合 UAP（⚠ 重评已失效，仅列相似思想供参考/ redo）
> 注：E13/E13b 重评显示 UAP 在 tiny / @50m / empty-FP 上**全面劣于 E12 基线**，
> 故该点暂不进入论文卖点。下列论文是"不确定性/置信融合"方向的同类工作，若后续 redo 可借鉴。

### ○ SUPER-Net — uncertainty propagation through segmentation network
- 把不确定性在分割网络中传播，用于置信校准/融合。

### ○ Knowledge-Guided Brain Tumor Seg — Unified Prior Fusion（能量相加）
- 多先验用"统一先验融合"以能量相加方式结合，与 U=|P_f−(1−P_b)| 思路相近但更鲁棒。

### ○ DST uncertainty-aware fusion of foundation + task models
- 用证据理论（DST）做基础模型与任务模型的不确定性感知融合。

### ○ 多模态 confidence-map late fusion
- 用各模态置信图做后期融合加权。

---

## 最该重点对比 / 引用的 3 篇（按贴合度）
1. **Niu et al. 2024 (FPE)** —— 创新点①最直接对手，用来凸显"我们做了因果消融而它没有"。
2. **UESDNet 2025 (ExpertSync + ego-info)** —— 创新点②"专家+高度"的同族，凸显我们级联两正交维度。
3. **SIFDAL 2024 (altitude-supervised scale disentangle)** —— 创新点②高度维度的最强佐证。

## 我们的差异化卖点（Related Work 落点）
- 大多数"先验/专家"论文只宣称"加先验有用"，**未排除容量/正则化解释**；
  BCP 用 E1(零)/E2(随机)≈E0 的因果对照链，证明增益只能来自先验语义内容。
- BCP 把先验作为**第4输入通道**（而非内部特征加权），且是**soft 连续置信**（非二值掩码）。
- 增益结构被量化揭示：集中于 tiny（89×）、随高度衰减、90m 失效——并据此用高度条件修复。
