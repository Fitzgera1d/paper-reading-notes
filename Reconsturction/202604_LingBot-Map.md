# Geometric Context Transformer for Streaming 3D Reconstruction - arXiv 2026

> Arxiv ID: [2604.14141](https://arxiv.org/abs/2604.14141)  
> Project: [https://technology.robbyant.com/lingbot-map](https://technology.robbyant.com/lingbot-map)  
> Code: [https://github.com/robbyant/lingbot-map](https://github.com/robbyant/lingbot-map)

## 一、问题定义与研究目标

这篇论文研究的是 streaming 3D reconstruction：给定连续到来的单目视频帧，模型需要在线恢复相机轨迹、深度图和可累积的三维点云。它接近 SLAM（simultaneous localization and mapping，即同步定位与建图）问题，但作者希望用前馈 3D foundation model 替代传统 SLAM 中手工设计的 keyframe、map、bundle adjustment 和 loop closure 组件，在保持实时性的同时获得长期几何一致性。

**核心任务是什么**：在时间 $t$，模型只能看到当前帧 $I_t$ 和历史帧 $\{I_1,\ldots,I_t\}$，不能使用未来帧。它要在 causal / streaming setting 下预测当前帧的 camera pose $\hat{P}_t$ 和 depth map $\hat{D}_t$，并通过连续帧的 pose-depth 累积得到全局 point cloud reconstruction。与离线 VGGT、DA3 这类 full-sequence feed-forward 模型不同，LingBot-Map 的核心约束是：输入序列可能非常长，推理必须边到边处理，memory 和 latency 不能随完整历史图像 token 线性爆炸。

**输入（Input）**：输入是连续 RGB 图像流 $\mathcal{I}=\{I_1,I_2,\ldots\}$。论文的默认实验分辨率为 $518\times378$；模型内部使用 ViT / DINOv2 风格 patch embedding，patch size 为 14，因此每帧被转换为 $M$ 个 image tokens。每帧 token 还会附加 camera token、4 个 register tokens 和 anchor / scale 相关 token，用于 pose prediction、trajectory memory 和尺度锚定。

**输出（Output）**：论文主输出是每帧 absolute camera pose $\hat{P}_t$ 和 depth map $\hat{D}_t$。pose 采用 camera-to-world transformation 监督；depth map 结合 pose 可反投影并累积成 point cloud。公开推理代码中还保留了 point head，能够输出 world points / confidence，但论文方法主线主要围绕 pose 与 depth 展开。

**任务的主要应用场景**：机器人导航、AR/VR、自动驾驶、长视频空间理解、embodied AI 的在线地图构建，以及任何需要从持续视觉流中实时维护三维空间记忆的系统。

**当前任务的关键挑战**：第一，长序列会产生 drift，早期小误差会沿轨迹传播。第二，保留完整历史 image tokens 会导致 KV cache、attention 计算和显存快速增长。第三，只保留很小的 recurrent state 又容易忘记关键几何上下文。第四，单目 streaming reconstruction 存在尺度与坐标系歧义，需要稳定的 reference frame。第五，局部位姿估计依赖密集视觉重叠，而长期一致性又需要全局历史线索，二者对 context 的需求不同。

**本文主要针对哪些挑战提出改进**：LingBot-Map 的核心目标是学习一种结构化但端到端的 streaming context：既像 SLAM 一样区分 reference、local window 和 global map，又不依赖手工优化。它提出 Geometric Context Attention（GCA），把上下文拆成 anchor context、local pose-reference window 和 trajectory memory，分别解决坐标/尺度锚定、局部密集几何匹配和长程 drift correction。

## 二、核心思想与主要贡献

论文的直观动机是：人类空间记忆不会保存每一帧视觉输入，而是选择性保留对导航和定位有用的结构线索。传统 SLAM 也有类似分工：参考帧提供坐标系，局部窗口用于相邻帧配准，全局地图用于长期一致性。LingBot-Map 将这种结构放进 Transformer attention mask 和 KV cache 设计中，让模型学习哪些上下文值得保留，而不是简单地保留全部历史或压缩成单一隐状态。

最相关的已有工作包括 VGGT、DA3、CUT3R、StreamVGGT、Stream3R、Wint3R、TTT3R 以及 VGGT-SLAM / MASt3R-SLAM。VGGT 和 DA3 是强大的离线 feed-forward 3D foundation models，但需要一次性访问完整输入集合；CUT3R 用 persistent recurrent state 做在线 3D perception，但强压缩容易遗忘；StreamVGGT / Stream3R / Wint3R 用 causal attention 或 cache 扩展 VGGT，但保留过多历史 token，长序列成本高；VGGT-SLAM / MASt3R-SLAM 等混合系统使用传统 SLAM 后端，但引入手工 heuristics 和迭代优化。

本文的关键差异是：它不是在离线模型外面加一个传统 SLAM 后处理，也不是把所有历史 token 全部缓存，而是设计一个几何结构化的 streaming attention。anchor frames 永久保留完整 tokens 以固定坐标和尺度；local window 保留最近 $k$ 帧完整 image tokens 以支持局部 dense matching；trajectory memory 对更早的历史只保留 6 个 context tokens / frame，以低成本提供长程轨迹线索。

本文主要贡献有三点。第一，提出 LingBot-Map / Geometric Context Transformer，用 GCA 同时建模 anchor context、pose-reference window 和 trajectory memory，在 10,000+ frames 的长序列上保持约 20 FPS 推理。第二，提出面向长序列的训练策略：先训练离线 base model，再把 global attention 替换为 GCA，并用 progressive view curriculum、relative pose loss 和 context parallelism 稳定训练。第三，构建较完整的 streaming 3D reconstruction benchmark，在 Oxford Spires、ETH3D、7-Scenes、Tanks and Temples、NRGBD 上证明其 pose estimation 和 point cloud reconstruction 均优于现有 streaming 方法，并在 Oxford Spires 上超过部分离线和优化型方法。

## 三、方法与实现细节（全文重点）

论文方法可以理解为“VGGT 风格前馈几何模型的 streaming 化”。普通 causal attention 能在线处理，但会缓存所有历史 image tokens；普通 sliding window 能限显存，但会丢失长程上下文。GCA 的核心是把不同几何作用的上下文分开存储和注意力访问，让模型在每帧推理时同时看到固定 reference、局部密集窗口和压缩轨迹记忆。

### 3.1 整体 Pipeline 概述

LingBot-Map 的完整流程分为训练和推理两层。训练时，作者先训练一个离线 base model，让 DINOv2 初始化的 ViT backbone 和 VGGT 风格的 alternating attention 学到可靠的多视图几何先验；第二阶段把 cross-frame global attention 替换为 GCA，并从 24 views 逐步扩展到 320 views，使模型适应长序列 causal reconstruction。推理时，前几帧作为 scale / anchor initialization，后续每帧被编码成 tokens，经过 Frame Attention 和 GCA 与三类上下文交互，camera head 输出 pose，depth head 输出 depth；对特别长的序列，Direct mode 可以换成 VO mode，用 overlapping windows 加 Sim(3) alignment 拼接局部轨迹。

### 3.2 端到端数据流

下表把单帧 streaming 推理的数据流展开。重点是区分完整 image tokens 和压缩 context tokens：前者提供 dense geometry，后者提供长期轨迹信息。

| 阶段 | 模块/操作 | 输入 | 输出 | 必要 shape / 维度 | 作用 |
|------|-----------|------|------|-------------------|------|
| 1 | 图像流输入 | 当前 RGB frame $I_t$ 与历史状态 | 当前帧图像 tensor | 默认实验 $518\times378$ | causal processing，只使用已观察帧 |
| 2 | DINOv2 / ViT patch embedding | $I_t$ | image tokens | 每帧 $M$ tokens，patch size 14 | 提取单帧视觉特征 |
| 3 | Special tokens 拼接 | image tokens | augmented frame tokens | camera token + 4 register tokens + anchor/scale token + $M$ image tokens | 给 pose、memory 和尺度锚定提供专用 token |
| 4 | Frame Attention | 当前帧 augmented tokens | refined per-frame tokens | 每帧内部 attention | 先做单帧空间特征 refinement |
| 5 | GCA: anchor context | 当前帧 tokens + anchor tokens | 坐标/尺度锚定后的 tokens | 最初 $n$ 帧保留完整 tokens | 固定全局 coordinate system 和 metric scale |
| 6 | GCA: local pose-reference window | 当前帧 tokens + 最近 $k$ 帧完整 tokens | 局部几何增强 tokens | window 内完整 image tokens | 提供密集局部重叠和相对位姿线索 |
| 7 | GCA: trajectory memory | 当前帧 tokens + 历史 compact tokens | 长程一致性增强 tokens | 6 context tokens / evicted frame | 保留完整轨迹历史，减少 drift |
| 8 | Camera head | camera token / aggregated tokens | pose encoding / $\hat P_t$ | per-frame camera pose | 预测当前帧 absolute camera pose |
| 9 | Depth head | image tokens / multi-layer features | $\hat D_t$ 与 uncertainty / confidence | dense depth map | 预测当前视角深度，用于 point cloud 累积 |
| 10 | Point cloud accumulation | $\hat P_t,\hat D_t$ | world-space point cloud | per-pixel back-projection | 构建持续更新的 3D reconstruction |

这个数据流的关键不是 head 本身，而是 GCA 如何控制 attention context。对于 $T$ 帧序列，GCA 保留 $n$ 个 anchor frames 的完整 tokens、$k$ 个 local window frames 的完整 tokens，以及其余历史帧的 6 个 context tokens。context 规模可写为：

$$
(n+k)M+6T.
$$

普通 causal attention 需要保留所有历史完整 tokens，规模为：

$$
T(M+6)=MT+6T.
$$

当 $M\approx500$ 时，每来一帧，causal attention 需要多存约 $M+6$ 个 tokens，而 GCA 只多存 6 个 compact tokens，增长率约降低 $80\times$。论文给出的例子是 $n=3,k=16,T=10{,}000$：causal attention 约累积 $5\times10^6$ tokens，而 GCA 约保留 $7\times10^4$ tokens。

### 3.3 关键模块逐个拆解

#### 3.3.1 Anchor Context：坐标与尺度锚定

单目 3D reconstruction 的绝对尺度和坐标系天然不确定。离线 VGGT 类方法可以用全局 point cloud 做 normalization，但 streaming inference 无法访问未来帧，因此 LingBot-Map 用最初 $n$ 帧作为 anchor frames。anchor frames 内部做 full attention，并引入 learnable anchor token，使模型能够识别这些帧的 reference role。初始化后，anchor frames 的 image tokens 和 context tokens 被保留，后续所有 streaming frames 都可以 attend to anchors。

训练时，作者用 anchor frames 的 ground-truth point cloud 定义规范化尺度。设 anchor point cloud 为 $\bar{\mathcal{X}}^{\text{anchor}}$，尺度因子为：

$$
s=\frac{1}{|\bar{\mathcal{X}}^{\text{anchor}}|}\sum_{\mathbf{x}\in\bar{\mathcal{X}}^{\text{anchor}}}\|\mathbf{x}\|_2.
$$

所有 ground-truth depths 和 camera translations 都除以 $s$。这个设计把训练监督和推理时的 reference frames 对齐：模型不是在任意尺度下漂移，而是从一开始就被约束到 anchor 定义的 coordinate / scale frame。

#### 3.3.2 Local Pose-Reference Window：局部密集几何上下文

anchor frames 只提供全局参考，但它们可能和当前帧相距很远，缺少足够重叠，无法支撑准确局部配准。因此 GCA 保留最近 $k$ 帧的完整 image tokens，形成 local pose-reference window。当前帧通过 GCA attend to 这些 dense visual tokens，获得相邻帧之间的视觉对应、局部结构和相对运动线索。

这个窗口也直接对应训练目标中的 relative pose loss。模型不仅要让每帧 pose 接近全局真值，还要让窗口内任意两帧的相对 pose 准确。这样做能减少局部小误差累积成长期 drift，尤其对 rotation 更重要：消融中去掉 relative loss 后，RPE-rot 从 2.26 变差到 5.35，ATE 从 7.46 变差到 8.25。

#### 3.3.3 Trajectory Memory：压缩的长期历史

只用 anchor + local window 仍然会丢掉中间历史。若当前帧离 anchor 很远，而局部窗口又只能看到最近帧，模型没有足够线索修正长期 drift。Trajectory memory 解决这个问题：对已经离开 anchor set 和 active window 的历史帧，模型丢弃昂贵的 $M$ 个 image tokens，只保留 camera token、anchor / scale token 和 4 个 register tokens，共 6 个 context tokens / frame。

这些 compact tokens 不保存完整图像细节，而是作为“轨迹摘要”保留每个历史观测对全局路径有用的信息。论文还加入 video temporal positional encodings，使这些 tokens 不只是无序历史集合，而带有时间顺序。消融显示 trajectory context tokens 能把 AUC@3 从 13.63 提升到 15.75，ATE 从 7.88 降到 7.46；加入 Video RoPE 后 ATE 进一步降到 5.98，是单项最大提升。

#### 3.3.4 Geometric Context Attention：结构化注意力掩码

GCA 可以看作一种几何先验约束下的 attention mask。它不是让当前帧 attend to 全部历史完整 tokens，而是让不同历史信息以不同粒度进入 attention：

- anchor frames：完整保留，用于 coordinate grounding 和 scale grounding。
- local window frames：完整保留，用于 dense local geometry 和 pose reference。
- evicted historical frames：只保留 compact context tokens，用于 long-range drift correction。

这对应 SLAM 中 reference frame、local map 和 global trajectory 的分工，但所有信息选择和融合都通过 Transformer attention 学习完成。与传统 SLAM 相比，它不需要显式 bundle adjustment；与普通 causal Transformer 相比，它不会把所有历史像素级 tokens 作为同等上下文。

#### 3.3.5 Geometric Context Transformer 网络结构

网络主体沿用 VGGT 风格的 alternating attention。每个输入图像先经过 DINOv2 初始化的 ViT backbone，生成 $M$ 个 image tokens；随后拼接 camera token $\mathbf{c}\in\mathbb{R}^C$、4 个 register tokens $\mathbf{r}_j\in\mathbb{R}^C$ 和 anchor / scale token。tokens 经过多层 Frame Attention 与 GCA 交替处理。Frame Attention 在单帧内部 refine image features；GCA 跨帧访问 anchor、local window 和 trajectory memory。

公开推理代码中的默认模型配置是 `embed_dim=1024`、`depth=24`、`num_heads=16`、`patch_size=14`、`num_register_tokens=4`，patch embedding 为 `dinov2_vitl14_reg`。depth head 是 DPT-style dense prediction head，输出 depth 与 confidence；camera head 对 camera-related tokens 做迭代 refinement，demo 默认迭代 4 次。

#### 3.3.6 KV Cache 与推理系统

推理系统借鉴 autoregressive LLM 的 KV cache：历史帧处理过后，其 key / value 状态被缓存，后续帧不用重复计算。但 naive causal KV cache 会随着完整历史增长。GCA 的 KV cache 更接近 paged memory layout：完整 patch/image tokens 只对 anchor 和 sliding window 保留，evicted frames 的 special/context tokens 长期保留。

论文实现基于 FlashInfer 的 paged KV-cache 和 sparse/paged attention kernels，避免反复 append / discard 时触发大块 contiguous memory reallocation。在 $518\times378$、1000 frames、window size 64 下，FlashInfer 版本约 20 FPS；对应 PyTorch contiguous KV-cache baseline 约 10.5 FPS。公开 demo 还提供 SDPA fallback，但默认推荐 FlashInfer。

### 3.4 损失函数与训练目标

LingBot-Map 的总损失由 depth、absolute pose 和 relative pose 三部分组成：

$$
\mathcal{L}
= \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}
+ \lambda_{\text{abs-pose}}\mathcal{L}_{\text{abs-pose}}
+ \lambda_{\text{rel-pose}}\mathcal{L}_{\text{rel-pose}}.
$$

**Depth loss** 继承 VGGT 的形式，同时约束 depth value、depth gradient 和 uncertainty：

$$
\mathcal{L}_{\text{depth}}
= \sum_{i=1}^{N}\left\|
\Sigma_i^D\odot(\hat D_i-D_i)
\right\|
+\left\|
\Sigma_i^D\odot(\nabla\hat D_i-\nabla D_i)
\right\|
-\alpha\log\Sigma_i^D.
$$

其中 $\hat D_i$ 是预测深度，$D_i$ 是真值深度，$\Sigma_i^D$ 是预测 uncertainty map，$\odot$ 表示逐元素乘法。depth residual 约束绝对几何，gradient residual 约束局部表面变化，uncertainty 项允许模型对困难区域降低过度惩罚，但也通过 $-\alpha\log\Sigma_i^D$ 防止无约束地提高 uncertainty。

**Absolute pose loss** 同样沿用 VGGT 的 Huber-style pose regression：

$$
\mathcal{L}_{\text{abs-pose}}
=\sum_{i=1}^{N}\left\|\hat{\mathbf{P}}_i-\mathbf{P}_i\right\|_{\epsilon}.
$$

与 VGGT 不同的是，本文监督 camera-to-world transformation，而不是 world-to-camera transformation。原因是 world-to-camera 参数化中 rotation 和 translation 更强耦合，长序列里 rotation error 会放大 translation estimation 的不稳定性。

**Relative pose loss** 作用于 local pose-reference window 内所有 frame pairs：

$$
\mathcal{L}_{\text{rel-pose}}
=\frac{1}{k(k-1)}
\sum_{\substack{i\ne j\\i,j\in\{1,\ldots,k\}}}
\left(
\mathcal{L}_{\text{rot}}(i,j)
+\lambda_{\text{trans}}\mathcal{L}_{\text{trans}}(i,j)
\right).
$$

$\mathcal{L}_{\text{rot}}(i,j)$ 是两帧 relative pose 的 geodesic rotation error，$\mathcal{L}_{\text{trans}}(i,j)$ 是 relative translation 的 $\ell_1$ error。这个 loss 是 causal 的，因为窗口中只包含已经观察过的帧。它和 absolute pose loss 互补：absolute loss 把每帧放到全局坐标中，relative loss 则约束相邻局部轨迹的一致性。

### 3.5 数据集与数据处理

训练数据由 29 个数据集组成，覆盖 indoor / outdoor、object-centric / scene-level、synthetic / real-world、multi-view collections / video sequences。Stage 1 使用更均衡的数据组合学习通用几何先验；Stage 2 提高长轨迹视频数据权重，以适应 streaming reconstruction。数据包括 BlendedMVS、HyperSim、MegaDepth、MVS Synth、GTA-SFM、CO3D、Objaverse、Texverse、Unreal4K、WildRGBD、TartanAir、TartanAirV2、TartanGround、Waymo、PointOdyssey、VirtualKITTI、Kubric、DL3DV、Replica、SceneRGBD、Mapfree、Aria Synthetic、ADT、ScanNet、ScanNet++、MatrixCity、MidAir、KITTI-360、Gibson、Matterport3D、HM3D，以及内部 game dataset。

论文把训练数据分成两类：multi-view collections 通常无明确时间顺序，可以用 nearby sampler 从空间邻域采样 2 到 24 frames；video sequences 有连续 camera trajectory，Stage 2 用 foldback video sampler 从长视频中采样 temporally coherent subsequences。foldback sampler 从随机帧开始按随机 stride 前进，到边界后反向并重新采样 stride，从而产生不同帧率、无单向时间偏置的训练片段。

附录的数据处理重点是统一异构数据源。不同数据集的 camera coordinate conventions 被转成统一 camera-to-world 表示；depth units 被统一为 meters；ScanNet / ScanNet++ 的 16-bit PNG depth 要除以 1000，VirtualKITTI2 的 centimeter depth 要除以 100；无效 depth、NaN / Inf、过远值、天空区域和几何 outliers 会被过滤或置零。所有数据被整理为统一 metadata，包括 scene list、frame index mapping、image/depth paths、intrinsics 和 $4\times4$ camera trajectories。

作者还额外构造了长程 traversal 数据。MatrixCity 的 aerial grid 和 street segments 被重组成连续 random-walk sequences；Gibson、Matterport3D、HM3D 通过 Habitat-Sim 渲染跨房间 RGB-D traversal videos，包含约 2,800 条序列、每条 1k 到 5k frames，总量约 14.4 TB。这些数据专门弥补现有室内数据缺少多房间长程穿越信号的问题。

评测数据集包括 Oxford Spires、ETH3D、7-Scenes、Tanks and Temples 和 NRGBD。pose 指标包括 relative pose error 的 AUC@3 / AUC@30、ATE、RPE-trans 和 RPE-rot；3D reconstruction 指标包括 Accuracy、Completeness 和 F1。重建评测中，预测点云和真值点云先用 Umeyama alignment，再做 ICP refinement。

### 3.6 训练流程、推理流程与复现备注

**Base model training**：第一阶段训练离线 base model。ViT backbone 从 DINOv2 初始化，patch size 为 14，模型包含 24 个 alternating blocks of frame attention and cross-frame attention。该阶段使用 global attention 而非 GCA，因为训练数据同时包含 unordered multi-view collections 和 temporally ordered videos；每个样本随机采样 2 到 24 views。优化器为 AdamW，base learning rate $2\times10^{-4}$，weight decay 0.05，前 5% steps 从 $10^{-8}$ linear warmup 到 base rate，剩余 95% cosine annealing 回 $10^{-8}$，总计 160K iterations。该阶段约需要 21,500 GPU hours。

**Streaming model training**：第二阶段从 base checkpoint 初始化，把 global attention 替换为 GCA。因为 GCA 的 query / key / value projections 与 global attention 同参数化，权重可以直接迁移。训练同样为 160K iterations，base learning rate $5\times10^{-4}$。view curriculum 从 24 views 线性增加到 320 views；local pose-reference window size $k$ 在 16 到 64 间随机采样。为了训练长序列，作者使用 Ulysses context parallelism，parallelism dimension 为 16，并基于 TorchTitan 和 Magi Attention 实现跨 GPU all-to-all attention。该阶段约需要 15,360 GPU hours。

**推理流程**：默认实验使用 Direct Output mode。模型先处理初始 scale / anchor frames，随后逐帧 causal inference；每一帧经过 DINOv2 patch embedding、Frame Attention、GCA、camera head 和 depth head，直接输出 absolute pose 与 depth。默认配置为 local pose-reference window $k=64$、keyframe interval $m=1$、resolution $518\times378$、bfloat16。论文认为 Direct mode 在 keyframe selection 下可稳定到约 3000 frames。

**长序列 VO mode**：当输入远超 Direct mode 的有效范围，例如上万帧，模型切换到 VO mode。输入被切成 overlapping local windows；每个窗口内的初始子集联合处理以建立局部 scale / coordinate system，后续帧继续 causal GCA；窗口结束后 reset state，并对相邻窗口 overlap 区域做 Sim(3) alignment，恢复相对 scale、rotation 和 translation。VO mode memory 有界，但窗口边界会引入额外 alignment drift，因此短中序列优先使用 Direct mode。

**复现备注**：公开仓库主要提供 inference / demo 代码和 checkpoint 加载逻辑，未包含完整训练脚本、训练 config 或 loss 权重数值。公开 demo 配置中，`image_size=518`、`patch_size=14`、`num_scale_frames=8`、`kv_cache_sliding_window=64`、`camera_num_iterations=4`；当 streaming 输入帧数超过 320 且未显式指定 `--keyframe_interval` 时，代码自动设为 `ceil(num_frames/320)`。demo 的 crop 预处理把 width 设为 518，并按 patch size 对 height 取整；若 height 超过 518，则 center crop 到 518。这些默认值足以复现推理 demo，但完整训练复现仍需要作者内部训练配置和大规模数据处理环境。

## 四、实验结果与有效性说明

实验结论可以分成三层：长序列 pose 是否稳定、pose 改善是否转化为更好重建、GCA 各组件是否必要。

**Oxford Spires pose estimation**：在 sparse setting（320 frames，每 12 帧采样）中，LingBot-Map 的 AUC@15 / AUC@30 / ATE 为 61.64 / 75.16 / 6.42。相比之下，最强离线 baseline DA3 为 49.84 / 56.68 / 12.87，优化型 VIPE 为 45.35 / 51.88 / 10.52，online CUT3R 为 5.98 / 14.95 / 18.16。这说明即使在 streaming setting 中，结构化几何上下文也能超过离线和优化型方法在复杂 campus-scale trajectory 上的泛化能力。

**长序列稳定性**：在 Oxford Spires dense setting（3,840 frames）中，LingBot-Map 的 ATE 从 sparse 的 6.42 只升到 7.11，增加 0.69；CUT3R 从 18.16 升到 32.47，Wint3R 从 21.10 升到 32.90。这个对比直接支持 trajectory memory 的设计目标：当序列长度扩大 12 倍时，模型仍能维持接近恒定的全局轨迹误差。

**跨数据集 pose generalization**：在 ETH3D、7-Scenes、Tanks and Temples 上，LingBot-Map 的 AUC@30 / ATE 分别为 86.20 / 0.22、78.59 / 0.08、92.80 / 0.20，均为表中最好。ETH3D 兼具室内外 laser-scanned depth，7-Scenes 有 motion blur、textureless surfaces 和 repetitive structures，Tanks and Temples 是大规模 outdoor captures；跨这些场景领先说明方法不是只对 Oxford Spires 过拟合。

**3D reconstruction quality**：在 ETH3D、7-Scenes、NRGBD 上，LingBot-Map 的 F1 分别为 98.98、80.39、64.26，均为表中最好。尤其 ETH3D 上，第二名 Wint3R 的 F1 为 77.28，差距超过 21 点；NRGBD 上第二名 Wint3R 为 56.96，LingBot-Map 提升到 64.26。由于 point cloud reconstruction 依赖 pose 和 depth，一致的轨迹估计直接减少了重复表面、错位墙面和碎裂结构。

**GCA 消融**：在 TartanGround 长序列消融中，仅有 relative loss 的 baseline AUC@3 / ATE 为 9.80 / 8.59；加入 anchor initialization 后变为 13.63 / 7.88；加入 context tokens 后变为 15.75 / 7.46；最终加入 Video RoPE 后达到 16.39 / 5.98。结果表明：anchor 解决尺度/坐标初始不稳定，context tokens 缓解长程 drift，Video RoPE 让 trajectory memory 真正具备时序结构。

**Window vs full causal attention**：window size 64 的 GCA 配置达到 ATE 5.98、RPE-trans 1.33、FPS 20.29、显存 13.28 GB；full causal attention 的 ATE 为 6.60、RPE-trans 1.50、FPS 11.87、显存 36.06 GB。这个结果有两个含义：第一，保留全部历史 image tokens 不仅慢，而且可能引入远处冗余或噪声；第二，GCA 的“完整局部 + 压缩全局”比全历史密集 attention 更适合 streaming reconstruction。

**局限性**：LingBot-Map 没有显式 loop closure，重访旧区域时仍可能无法像传统 SLAM 那样做闭环校正；trajectory memory 每帧压缩为固定数量 tokens，可能损失极长序列中的细粒度几何；方法不做 test-time optimization，因此在极难场景下仍缺少 BA-like refinement。作者未来方向包括把 bundle-adjustment-like refinement 和 loop closure 融入 attention，引入 LiDAR / IMU 等多模态输入，以及扩展到动态场景和下游 navigation / novel view synthesis。
