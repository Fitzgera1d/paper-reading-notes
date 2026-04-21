# CV Paper Reading Notes

中文 | [English](./README_EN.md)

## 阅读什么论文？如何阅读论文？

### 1. 如何发现领域相关的好论文？

- [Scholar](https://www.scholar-inbox.com/)
- [HuggingFace](https://huggingface.co/papers/)
- [alphaXiv](https://www.alphaxiv.org/)

### 2. 如何阅读一篇CV领域的论文？

- 核心理念：分层次、有目的地进行多遍阅读
- 阅读前的准备：明确目标 (Reading Goal)
  - 目标A：了解大意/文献综述 (Get the Gist)
    - 场景：你需要快速了解某个领域有哪些工作，或者判断这篇论文是否与你的研究相关。
    - 深度：浅层阅读，重点是了解其解决了什么问题和核心思想。
  - 目标B：理解核心方法 (Understand the Method)
    - 场景：这篇论文与你的工作高度相关，你希望借鉴或复现它的方法。
    - 深度：中层阅读，重点是搞清楚模型架构、实验设置和关键结果。
  - 目标C：深入批判与启发 (Deep Critique & Inspiration)
    - 场景：这是你研究领域的核心论文，你需要对它进行深入分析，找到其优点、缺点，并为自己的研究寻找创新点。
    - 深度：深层阅读，需要理解所有细节、挑战作者的假设、思考改进方案。
- <details>
  <summary>阅读方法：三遍阅读法 (The Three-Pass Approach)</summary>

  - 第一遍：鸟瞰全局，建立框架 (5-10分钟)

    - 目标：快速判断论文的相关性，并建立对文章的宏观理解。**只关注“是什么”和“好不好”**。
    - 主次顺序与阅读内容：
      1. **标题 (Title)**：快速了解主题。
      2. **摘要 (Abstract)**：这是全文的精华。仔细阅读，找几个答案：
         1. 背景 (Context)：这是什么领域的问题？
         2. 问题 (Problem**)**：他们具体解决了什么痛点？
         3. 方法 (Method)：他们提出了什么核心方法？
         4. 结果 (Results)：他们的结果有多好？
      3. **图和表 (Figures & Tables)**：**这是 CV 论文的灵魂**
         1. 优先看架构图 (Architecture Figure)：它比文字更直观地展示了模型的结构。
         2. 再看结果表 (Results Table)：找到主要的结果表格，关注他们的方法（通常是加粗的那一行）和 SOTA (State-of-the-Art) 或关键基线 (Baseline) 的对比。指标是提升了还是下降了？
         3. 最后看效果图 (Qualitative Results)：看看他们生成的图像、检测框或分割结果，直观感受效果好坏。
      4. 引言 (Introduction)：快速浏览，重点阅读**最后一段**。作者通常会在这里明确总结本文的贡献点 (Contributions)。
      5. 结论 (Conclusion)：快速阅读，它会重申论文的核心思想和成果，并指出未来的研究方向。
    - 第一遍读完后，你应该能回答：
      - 这篇论文解决了什么问题？
      - 它的核心思想是什么？
      - 它的主要成果是什么？
      - 这篇论文值得我花更多时间吗？

  - 第二遍：深入细节，理解方法 (约1小时)

    - 目标：搞清楚方法的具体实现细节。**重点关注“怎么做”和“怎么评”**。
    - 主次顺序与阅读内容：
      1. **方法论 (Methodology / Approach)**：这是本轮的阅读核心。
         1. 仔细重读架构图，并与正文描述进行对应，确保你理解数据是如何在模型中流动的。
         2. 关注关键模块：作者通常会详细介绍他们提出的创新点（例如，一个新的注意力模块、一个新的损失函数）。理解这些模块的输入、输出和内部计算逻辑。
         3. 标记不懂的术语和公式：暂时不要卡在复杂的数学推导上，先标记下来，尝试理解公式的**物理意义**（目的）。
      2. **实验设置 (Experiments / Implementation Details)**：
         1. 数据集 (Datasets)：他们用了哪些公开数据集？（如 ImageNet, COCO）
         2. 评估指标 (Evaluation Metrics)：他们用什么指标来衡量好坏？（如 mAP, IoU, PSNR）
         3. 基线模型 (Baselines)：他们主要和谁进行比较？
         4. 训练细节：学习率、优化器、数据增强等。这些信息对于复现至关重要。
      3. **消融实验 (Ablation Studies)**：**这是 CV 论文中极其重要的部分！**
         1. 作者会通过“控制变量”的方式，逐一移除或替换他们提出的新模块，以证明**每个创新点都是有效的**。仔细阅读这部分，可以让你深刻理解作者的设计思路和每个组件的贡献。
    - 第二遍读完后，你应该能：
      - 向别人清晰地解释这个模型的工作原理。
      - 理解作者是如何通过实验来证明他们的方法是有效的。
      - 对如何复现这篇论文有一个大致的规划。

  - 第三遍：批判性思考，“虚拟复现” (1-N小时)

    - 目标：达到专家级的理解，能够批判性地评估该工作，并从中获得启发。**重点关注“为什么”和“如果”**。
    - 主次顺序与阅读内容：
      1. 深入推导数学细节：回到第二遍中标记的公式，尝试从头到尾推导一遍。确保你理解其中的每一个假设。
      2. 批判性地审视方法和实验：
         1. 提问“为什么”：作者**为什么**要这么设计？有没有其他更好的选择？
         2. 寻找“漏洞”：实验对比是否公平？有没有“炼丹”的嫌疑？他们报告的指标是否掩盖了某些方面的性能下降？有没有他们刻意回避比较的强大基线？
         3. 思考局限性 (Limitations)：这个方法在什么情况下可能会失效？
      3. 精读相关工作 (Related Work)：
         1. 在理解了本文工作后，再回过头去精读相关工作部分。这能帮助你更清晰地定位这篇论文在整个领域知识图谱中的位置，理解它的**真正创新点**在哪里。
      4. 寻找启发点 (Inspiration)：
         1. 提问“如果”：**如果**我把这个模块用到我自己的任务中，会怎么样？**如果**我把 A 方法和 B 方法的优点结合起来，会怎么样？
         2. 关注结论中提到的**未来工作 (Future Work)**，这通常是很好的研究方向。
    - 第三遍读完后，你应该能：
      - 像审稿人一样，对这篇论文的优点和缺点进行全面评价。
      - 产生新的研究想法，或者知道如何将它应用到你的项目中。
      - （理想情况下）能够着手复现论文的核心代码。

  </details>

- 贯穿始终的阅读目标：
  - 论文完成了什么样的task：输入是什么，输出是什么？任务的难点是什么，为什么会成为难点？
  - 论文设计了什么样的pipeline：模型架构是什么样的，损失函数是怎么设计的？
  - 论文好在哪里，为什么好：设计的架构优势在哪里，为什么会有这样的优势？是怎么解决难点的？

### 使用什么样的提示词让AI帮你总结分析论文？

- 工具链接：[alphaXiv](https://www.alphaxiv.org/)
- 使用模型：gemini-2.5-pro
- 简单提示词：
  ```markdown
  1. 本研究的直观动机是什么？它在技术设计中是如何体现的？
  2. 本研究与哪项已有工作最为相关，在哪些关键方面进行了改进或创新？
  ```
- [复杂提示词](./.prompts/cv_paper_notes_cn.md)：
  ```markdown
  # 提示词 Prompts
  
  **角色设定**：你是一位资深的计算机图形学研究员，擅长对论文进行技术拆解，并在必要时结合代码实现补全论文中未充分展开的细节。你的目标不是生成泛泛而谈的综述，而是输出一份**论文优先、细节扎实、结构清楚、可复查**的研究笔记，尤其要把方法部分讲透。
  
  ## 核心原则
  
  以下原则用于约束整份笔记的分析顺序、信息来源和表达方式。
  
  1. **论文优先，代码仅作定向补全**
     - 分析主线应建立在论文正文、图示、公式和附录之上。
     - 代码仓库不是分析重点，只在以下情况下作为辅助证据源使用：
       - 网络结构描述不清
       - 模块连接关系不明确
       - shape / 维度变化未写清
       - 损失函数实现细节缺失
       - 训练或推理流程存在省略
       - 数据预处理或后处理步骤描述不足
     - 不需要系统介绍整个代码仓库，也不需要遍历式讲解项目结构。只有在论文存在模糊点时，才去代码中做问题驱动的定向查证。
     - 若代码与论文不一致，必须明确指出；若代码也无法确认，则直接写“未确认”，不要推测。
  
  2. **正确性优先于文风**
     - 若“叙述流畅”和“技术准确”冲突，优先保证技术准确。
     - 方法论部分允许使用表格、清单、shape trace 和模块分解，不要为了行文连贯而牺牲细节。
     - 严禁使用“若干层”“多个模块”“通过 backbone 提取特征”等模糊表述。应尽可能写清层级、模块、输入输出、shape 变化和中间表示。
  
  3. **区分信息来源**
     - 对关键技术结论尽量标注来源：
       - `[Paper]`：论文明确写出
       - `[Code]`：代码中可直接确认
       - `[Inference]`：根据论文和代码做出的合理推断
     - 不确定的内容必须显式说明，不要将推断写成事实。
  
  4. **方法论是全文重点**
     - 全文应以“任务定义清楚、方法拆解透彻、实验结论克制”为目标。
     - 方法与实现细节部分应占全文主体篇幅，重点回答：
       - 输入是什么，输出是什么
       - 数据如何流动
       - 关键模块如何设计
       - shape 如何变化
       - 损失如何构成
       - 训练与推理如何执行
       - 代码补全了哪些论文没有说清的细节
  
  ## 输出结构要求
  
  请严格按照以下结构组织最终笔记，并把篇幅重点放在方法部分。
  
  ---
  
  # [论文标题] - [会议/期刊简称_年份]
  
  > [Arxiv ID / 论文链接]
  
  ## 一、问题定义与研究目标
  
  先用较完整的叙述说明论文所研究的问题、背景和目标。重点回答以下内容：
  
  - **核心任务是什么**
  - **输入（Input）**
    - 输入数据类型与组织形式
    - 输入 shape / 维度表示
    - 是否包含多视角图像、点云、相机参数、姿态参数、文本条件、latent code 等附加条件
  - **输出（Output）**
    - 输出数据类型与具体含义
    - 输出 shape / 维度表示
  - **任务的主要应用场景**
  - **当前任务的关键挑战**
  - **本文主要针对哪些挑战提出改进**
  
  这一部分要帮助读者先建立问题空间，但不要展开过多泛泛背景。
  
  ## 二、核心思想与主要贡献
  
  这一部分聚焦“作者到底提出了什么、为什么这样做”，要求简洁但具体。
  
  - 论文的直观动机是什么
  - 本文最相关的已有工作有哪些
  - 本文相较于这些工作的关键差异是什么
  - 本文最重要的 1-3 个贡献分别是什么
  
  如果论文的方法优势能够直接从设计动机和贡献中解释清楚，也请在这一部分一并说明，避免和后文重复。
  
  ## 三、方法与实现细节（全文重点）
  
  > 这一部分必须是全文核心。请不要只做概括，必须沿着“输入 -> 中间表示 -> 输出”的主线把方法拆开讲清楚。
  
  ### 3.1 整体 Pipeline 概述
  
  先用一段较完整的话概括整个方法的主要阶段，以及训练和推理的大致流程。
  
  ### 3.2 端到端数据流
  
  从原始输入开始，逐步说明数据如何流经各个模块直到最终输出。必须写清：
  
  - 每一步的输入和输出
  - 关键中间变量及其含义
  - shape / 维度如何变化
  - 是否有分支、融合、skip connection、迭代优化或渲染过程
  
  如有必要，可以用表格辅助：
  
  | 阶段 | 模块/操作 | 输入 | 输出 | Shape 变化 | 作用 | 来源 |
  |------|-----------|------|------|------------|------|------|
  
  ### 3.3 关键模块逐个拆解
  
  对于每个关键模块，都尽量写清以下内容：
  
  - 模块名称
  - 它在整体 pipeline 中的位置与作用
  - 输入、输出及其语义
  - 内部由哪些层或子模块组成
  - 关键设计细节
    - 卷积核大小、步长、通道数、激活函数、归一化方式
    - attention / transformer 的 hidden dim、head 数、token 组织方式
    - positional encoding、feature fusion、skip connection 的具体做法
    - 参数是否共享
    - 是否存在多阶段 / 迭代 refinement
  - 数据经过该模块后的 shape 变化
  - 该模块解决了什么问题
  - 如果论文有消融实验，结合结果说明该模块的贡献
  
  如果论文没有写清，而代码能补全，请用 `[Code]` 标注；如果只能合理推断，请用 `[Inference]` 标注。
  
  ### 3.4 损失函数与训练目标
  
  详细说明：
  
  - 总损失由哪些部分组成
  - 每个损失项的数学形式
  - 每项损失约束的对象是什么
  - 各损失项的权重或调度方式
  - 哪些内容是论文明确写出的，哪些是代码补全的
  - 论文是否通过消融或分析证明这些损失的作用
  
  请尽量使用 LaTeX 表示公式，并解释变量含义。
  
  ### 3.5 数据集与数据处理
  
  写清：
  
  - 使用了哪些训练 / 验证 / 测试数据集
  - 一个样本是如何构造的
  - 是否有裁剪、采样、归一化、增强、重建、重投影、mesh / pose 预处理等步骤
  - 这些处理步骤中，哪些是论文明确说明的，哪些是从代码补全得到的
  
  ### 3.6 训练流程、推理流程与代码补全说明
  
  这一部分专门用于避免“论文说得顺，细节却落不下来”的问题。请明确写出：
  
  - 训练阶段的完整流程
  - 推理阶段的完整流程
  - 两者是否存在差异
  - 是否有 test-time optimization、pretrain / fine-tune、多阶段训练等策略
  - 论文中哪些地方描述不清
  - 代码具体补全了哪些实现细节
  - 若论文与代码不一致，请明确列出
  
  ## 四、实验结果与有效性说明
  
  这一部分保持克制，不需要铺陈太多。
  
  - 简要总结最关键的实验结果
  - 说明这些结果证明了什么
  - 若有消融实验，挑最关键的结果解释各模块或损失项的必要性
  - 若有代表性可视化结果，简要说明它们体现了模型的哪些能力或局限
  
  **不要编造任何数据。**
  如果需要引用结果，请严格使用论文中的原始数值。
  不必强制生成完整表格，可直接围绕论文中的核心结果进行叙述性解读。
  
  ---
  
  ## 专业术语与表达要求
  
  这些要求用于统一术语解释、公式表达和整体可读性。
  
  - 对计算机图形学和视觉领域的专业术语，在首次出现时给出简要解释。
  - 数学公式尽量使用 LaTeX，不要使用反引号包裹公式。
  - 对符号进行解释，不要只写公式不给变量含义。
  
  ## 最终要求
  
  生成结果时，请同时满足以下输出约束。
  
  - 输出必须是 `.md` 文件内容
  - 最终结果放在 Markdown code block 中展示
  - 重点是**把方法讲清楚、讲具体、讲到可以复查**
  - 代码只用于补全论文没有讲清的地方，不要喧宾夺主
  - 若某些细节无法确认，请明确写“未确认”，不要模糊带过

  ```

- 流程图绘制：

  ```markdown
  请使用mermaid绘制论文中*网络架构与数据流*的流程图。
  ```

## 笔记目录

- **Base/**
  - [202005_RAFT](Base/202005_RAFT.md)
  - **AE/**
    - [202204_MultiMAE](Base/AE/202204_MultiMAE.md)
    - [202510_RAEs](Base/AE/202510_RAEs.md)
  - **Diffusion_Models/**
    - [000000_survey](Base/Diffusion_Models/000000_survey.md)
    - [201511_DMs](Base/Diffusion_Models/201511_DMs.md)
    - [202012_DDPMs](Base/Diffusion_Models/202012_DDPMs.md)
    - [202106_ADMs](Base/Diffusion_Models/202106_ADMs.md)
    - [202201_EDMs](Base/Diffusion_Models/202201_EDMs.md)
    - [202204_LDMs](Base/Diffusion_Models/202204_LDMs.md)
    - [202305_DiT](Base/Diffusion_Models/202305_DiT.md)
    - [202312_CrossDiT](Base/Diffusion_Models/202312_CrossDiT.md)
    - [202312_UViT](Base/Diffusion_Models/202312_UViT.md)
    - [202405_MMDiT](Base/Diffusion_Models/202405_MMDiT.md)
    - [202501_FLUX](Base/Diffusion_Models/202501_FLUX.md)
  - **Tokenization/**
    - [000000_survey](Base/Tokenization/000000_survey.md)
    - [202201_iBOT](Base/Tokenization/202201_iBOT.md)
    - [202402_DINOv2](Base/Tokenization/202402_DINOv2.md)
    - [202504_PE](Base/Tokenization/202504_PE.md)
    - [202508_DINOv3](Base/Tokenization/202508_DINOv3.md)
  - **Transformer/**
    - [202110_LoRA](Base/Transformer/202110_LoRA.md)
    - [202306_Hiera](Base/Transformer/202306_Hiera.md)
- **Detection/**
  - [202308_PlainDETR](Detection/202308_PlainDETR.md)
  - [202410_SAM2](Detection/202410_SAM2.md)
  - [202412_CubifyAnything](Detection/202412_CubifyAnything.md)
  - [202511_SAM3](Detection/202511_SAM3.md)
- **Generation/**
  - [202412_Imagen3](Generation/202412_Imagen3.md)
- **Human_Motion/**
  - **Monocular/**
    - [202306_SAHMR](Human_Motion/Monocular/202306_SAHMR.md)
    - [202308_HMR2.0](Human_Motion/Monocular/202308_HMR2.0.md)
    - [202312_ViTPose++](Human_Motion/Monocular/202312_ViTPose++.md)
    - [202403_TRAM](Human_Motion/Monocular/202403_TRAM.md)
    - [202407_MultiHMR](Human_Motion/Monocular/202407_MultiHMR.md)
    - [202409_GVHMR](Human_Motion/Monocular/202409_GVHMR.md)
    - [202411_CameraHMR](Human_Motion/Monocular/202411_CameraHMR.md)
    - [202411_SATHMR](Human_Motion/Monocular/202411_SATHMR.md)
    - [202501_JOSH](Human_Motion/Monocular/202501_JOSH.md)
    - [202503_EDPose++](Human_Motion/Monocular/202503_EDPose++.md)
    - [202504_PromptHMR](Human_Motion/Monocular/202504_PromptHMR.md)
    - [202510_HUMAN3R](Human_Motion/Monocular/202510_HUMAN3R.md)
    - [202510_PhySIC](Human_Motion/Monocular/202510_PhySIC.md)
    - [202511_3DB](Human_Motion/Monocular/202511_3DB.md)
    - [202601_UniSH](Human_Motion/Monocular/202601_UniSH.md)
  - **MultiView/**
    - [202111_MvP](Human_Motion/MultiView/202111_MvP.md)
    - [202208_QuickPose](Human_Motion/MultiView/202208_QuickPose.md)
    - [202505_HSfM](Human_Motion/MultiView/202505_HSfM.md)
    - [202508_HAMSt3R](Human_Motion/MultiView/202508_HAMSt3R.md)
    - [202509_HART](Human_Motion/MultiView/202509_HART.md)
    - [202510_Kineo](Human_Motion/MultiView/202510_Kineo.md)
    - [202601_DiffProxy](Human_Motion/MultiView/202601_DiffProxy.md)
    - [202603_DuoMo](Human_Motion/MultiView/202603_DuoMo.md)
- **Reconsturction/**
  - [202501_CUT3R](Reconsturction/202501_CUT3R.md)
  - [202503_VGGT](Reconsturction/202503_VGGT.md)
  - [202512_C3G](Reconsturction/202512_C3G.md)
  - [202601_MoE3D](Reconsturction/202601_MoE3D.md)
  - **CAD/**
    - [202503_LiteReality](Reconsturction/CAD/202503_LiteReality.md)
    - [202505_CAST](Reconsturction/CAD/202505_CAST.md)
    - [202511_SAM3D](Reconsturction/CAD/202511_SAM3D.md)
    - **Ref/**
      - [202405_BlockFusion](Reconsturction/CAD/Ref/202405_BlockFusion.md)
      - [202411_BrepGen](Reconsturction/CAD/Ref/202411_BrepGen.md)
  - **Depth_Estimation/**
    - [202105_DPT](Reconsturction/Depth_Estimation/202105_DPT.md)
    - [202410_DA2](Reconsturction/Depth_Estimation/202410_DA2.md)
    - [202504_MoGe](Reconsturction/Depth_Estimation/202504_MoGe.md)
    - [202507_MoGe2](Reconsturction/Depth_Estimation/202507_MoGe2.md)
    - [202509_MapAnything](Reconsturction/Depth_Estimation/202509_MapAnything.md)
    - [202511_DA3](Reconsturction/Depth_Estimation/202511_DA3.md)
    - [202512_D4RT](Reconsturction/Depth_Estimation/202512_D4RT.md)
  - **Matching/**
    - [202305_RoMa](Reconsturction/Matching/202305_RoMa.md)
    - [202511_RoMa2](Reconsturction/Matching/202511_RoMa2.md)