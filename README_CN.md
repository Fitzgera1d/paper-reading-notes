# CV Paper Reading Notes

[English](./README_EN.md) | 中文

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
  # 提示词Prompts
  
  **角色设定**：你是一位资深的计算机图形学研究员，精通从学术论文中提炼深层见解，并能以高度连贯和逻辑严谨的语言组织笔记。你的目标是生成一份不仅信息全面，而且易于理解和回顾的深度研究笔记。
  
  ## 核心处理原则
  1.  **追求连贯性与逻辑性**：在生成笔记时，请尽量使用流畅的叙述性语言，减少不必要的分点和孤立的段落。确保各个分析模块之间有自然的过渡和逻辑关联，形成一篇浑然一体的深度解读。
  2.  **分层精读与信息整合**：
      *   **速览与框架构建 (10分钟)**：快速阅读摘要、引言、结论，理解论文核心贡献和大致脉络。
      *   **细节深潜 (核心时间)**：精读方法论、实验设计、网络结构、损失函数等关键章节。
      *   **批判性思考与提炼**：标注创新点、潜在局限，并思考其深层原理和潜在影响。
  
  ## 笔记结构与具体要求
  请围绕以下结构组织笔记，确保每个部分的阐述都详尽且连贯：
  
  ---
  
  # 论文标题: [论文标题] - [会议/期刊] [年份]
  
  ### 一、引言与核心问题
  
  *   简述论文的研究背景和重要性。
  *   **论文试图解决的核心任务是什么？**
       *   **输入 (Input)**: 详细描述输入数据的类型和具体形态。例如，是单张/多张图像、点云、文本描述还是其他？请明确指出其**数据维度/Shape**（例如，对于图像是 `[Batch_size, Channels, Height, Width]`，对于点云是 `[Batch_size, Num_points, Feature_dim]`）。如果输入设计有特殊之处（如特定的编码方式、多模态输入融合等），请有条理地展开介绍。
       *   **输出 (Output)**: 详细描述输出数据的类型和具体形态，同样明确其**数据维度/Shape**。例如，是生成图像、3D模型、分割掩码、变换参数还是其他？
       *   **任务的应用场景**: 列举该任务在计算机图形学或其他相关领域中的典型应用场景。
       *   **当前任务的挑战 (Pain Points)**: 深入分析当前任务面临的主要难点是什么？为什么这些会成为难点（例如，计算复杂度高、数据稀疏性、真实感不足、泛化能力弱等）？
       *   **论文针对的难点**: 明确指出这篇论文主要聚焦于上述哪些难点进行设计和改进。
  
  ### 二、核心思想与主要贡献
  
  *   **直观动机与设计体现**: 本研究的直观动机是什么？这一动机是如何体现在论文的技术设计中的？
  *   **与相关工作的比较与创新**: 本研究与哪（几）项已有工作最为相关？在哪些关键方面对这些相关工作进行了改进或提出了新的思路？
  *   **核心贡献与创新点**: 清晰提炼论文最主要的1-3个核心贡献与创新点。
  
  ### 三、论文方法论 (The Proposed Pipeline)
  
     *   **整体架构概述**: 首先用一段连贯的文字概括描述论文提出的方法/模型的整体Pipeline。
     *   **详细网络架构与数据流**:
          *   从**数据预处理**开始，逐步描述数据是如何在网络中流动的，直至最终输出。
          *   **逐层/逐模块解析**: 详细描述网络中每个关键层或模块的设计。包括但不限于：
              *   **层/模块类型**: (例如，Convolutional Layer, Transformer Encoder, Attention Mechanism, Custom-designed Block等)。
              *   **设计细节**: 该层/模块的具体配置（如卷积核大小、步长、通道数变化、激活函数、归一化方式等）。如果某个模块设计新颖或至关重要，请详细解释其内部结构和工作原理。
              *   **形状变换 (Shape Transformation)**: 清晰说明数据在经过每一层/模块后的**形状 (Shape) 或维度变化**。
              *   **中间变量**: 描述重要的中间特征表示及其意义。
              *   **结合消融实验的作用分析**: 如果论文包含消融实验，请结合实验结果阐述每个关键层/模块或数据处理步骤对最终性能的具体贡献和作用。
     *   **损失函数 (Loss Function)**:
          *   **设计理念**: 详细描述损失函数的构成。它包含哪些部分？每个部分的数学形式是什么？
          *   **关注重点**: 该损失函数主要关注哪些方面的信息或约束（例如，像素级重建、感知相似性、结构一致性、对抗性损失等）？
          *   **训练实施**: 在训练过程中，这个损失函数是如何被应用的？是否有特殊的加权或调度策略？
          *   **对性能的贡献**: 论文中是否有证据或分析表明该损失函数设计对最终性能的具体贡献？
     *   **数据集 (Dataset)**:
          *   **所用数据集**: 明确指出论文在训练和评估中使用了哪些公开或私有数据集。
          *   **特殊处理**: 论文是否对数据集进行了特殊的预处理、增强 (Data Augmentation)、筛选或构建？如有，请描述具体方法及其声明的目的或效果。
  
  ### 四、实验结果与分析
  
     *   **核心实验结果**: 简要总结主要的实验结果，用连贯的语言解读这些结果。可使用论文中的关键对比表格辅助说明，但重点在于解读。请严格使用论文中的数据，严禁编造数据。
  
          | 指标     | 基线方法A | 基线方法B | 本文方法 | 提升幅度 |
          |----------|-----------|-----------|----------|---------|
          | [Metric 1] | ...       | ...       | ...      | ...     |
          | [Metric 2] | ...       | ...       | ...      | ...     |
  
     *   **消融研究解读**: 若论文包含消融实验，详细解读其结果，阐明模型各组成部分的必要性和贡献。
     *   **可视化结果分析**: 如果有，描述并分析论文中具有代表性的可视化结果，它们如何证明方法的有效性。
  
  ### 五、方法优势与深层分析
  
     *   **架构/设计优势**:
          *   **优势详述**: 结合前述的网络架构、损失函数等具体实现细节，深入分析论文提出的方法为什么具有优势。例如，是因为更高效的特征提取、更鲁棒的优化目标、对特定数据模式的更好适应，还是其他原因？
          *   **原理阐释**: 解释这些设计为什么能带来这样的优势，背后的原理是什么？
     *   **解决难点的思想与实践**: 总结论文是通过一种什么样的核心思想，并如何通过具体的模型设计、训练策略等手段，在实践中有效解决其针对的核心难点的。
  
  ### 六、结论与个人思考 (可选，但推荐)
  
     *   论文的主要结论回顾。
     *   **潜在局限性**: 基于你的理解，指出该方法可能存在的局限性或未解决的问题。
     *   **未来工作方向**: 提出可能的改进方向或基于此工作的未来研究思路。
     *   **对个人研究的启发**: (如果适用) 这篇论文对你自己的研究有何启发？
  
  ### 七、代码参考与分析建议 (若GitHub仓库可访问)
  
     *   **仓库链接**: [如果找到，请提供]
     *   **核心模块实现探讨**: 基于论文描述，建议关注代码库中哪些核心模块（例如，新颖的网络层、关键的算法流程）的实现。如果时间允许且代码可读性高，可简要讨论代码实现与论文描述的一致性或关键实现技巧。若不进行详细代码分析，则建议读者自行查阅并关注特定模块，例如："建议读者查阅作者提供的代码[链接]，重点关注[模块A]和[模块B]的实现，以理解其具体工作方式和参数配置。"
  
  ---
  
  **专业术语处理**:
  *   对于计算机图形学领域的专业术语（如BRDF, Monte Carlo Path Tracing, SDF, NeRF, Voxelization, Mesh Processing等），请在首次出现时给出简要清晰的解释或注解。
  *   数学公式及数学符号尽量使用LaTeX格式呈现（不要使用"`"），并标注关键变量的含义。
  
  **请严格按照上述结构和要求，对这篇论文进行深度解读，并生成一份连贯、详实、富有洞察力的研究笔记。**
  
  最后，请将笔记输出到成.md文件格式，并在code block中显示出来。
  ```

- 流程图绘制：

  ```markdown
  请使用mermaid绘制论文中*网络架构与数据流*的流程图。
  ```

## 笔记目录

- **base/**
  - [RAFT](base/RAFT.md)
  - **diffusion/**
    - [DDPMs](base/diffusion/DDPMs.md)
    - [DMs](base/diffusion/DMs.md)
    - [LDMs](base/diffusion/LDMs.md)
    - [survey](base/diffusion/survey.md)
  - **ssm/**
  - **tokenizer/**
    - [DINOv2](base/tokenizer/DINOv2.md)
    - [iBOT](base/tokenizer/iBOT.md)
    - [survey](base/tokenizer/survey.md)
  - **transformer/**
- **motion/**
  - [202208_QuickPose](motion/202208_QuickPose.md)
  - [202306_HMR3D](motion/202306_HMR3D.md)
  - [202308_HMR2.0](motion/202308_HMR2.0.md)
  - [202312_ViTPose++](motion/202312_ViTPose++.md)
  - [202403_TRAM](motion/202403_TRAM.md)
  - [202409_GVHMR](motion/202409_GVHMR.md)
  - [202501_JOSH](motion/202501_JOSH.md)
- **reconstruction/**
  - [VGGT](reconstruction/VGGT.md)
  - **depth/**
    - [202105_DPT](reconstruction/depth/202105_DPT.md)
    - [202410_DepthAnythingV2](reconstruction/depth/202410_DepthAnythingV2.md)
  - **match/**
    - [RoMa](reconstruction/match/RoMa.md)