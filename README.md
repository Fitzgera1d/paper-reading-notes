# CV Paper Reading Notes

[中文](./README_CN.md) | English

## What Papers to Read? How to Read Them?

### 1. How to Discover Good Papers in a Field?

- [Scholar](https://www.scholar-inbox.com/)
- [HuggingFace](https://huggingface.co/papers/)
- [alphaXiv](https://www.alphaxiv.org/)

### 2. How to Read a Paper in the CV Field?

- Core Philosophy: Conduct multiple readings in a hierarchical and purposeful manner.
- Preparation Before Reading: Clarify Your Reading Goal
  - Goal A: Get the Gist / Literature Review
    - Scenario: You need to quickly understand the work in a certain area or determine if a paper is relevant to your research.
    - Depth: Skim reading, focusing on understanding the problem it solves and its core idea.
  - Goal B: Understand the Method
    - Scenario: The paper is highly relevant to your work, and you want to adopt or reproduce its method.
    - Depth: Medium-level reading, focusing on understanding the model architecture, experimental setup, and key results.
  - Goal C: Deep Critique & Inspiration
    - Scenario: This is a core paper in your research field. You need to analyze it in depth, find its strengths and weaknesses, and seek inspiration for your own research.
    - Depth: Deep reading, requiring an understanding of all details, challenging the author's assumptions, and thinking about improvement plans.
- <details>
  <summary>Reading Method: The Three-Pass Approach</summary>

  - **First Pass: Get a Bird's-Eye View, Build a Framework (5-10 minutes)**

    - Goal: Quickly assess the paper's relevance and build a high-level understanding. **Focus only on "what it is" and "how good it is."**
    - Reading Order and Content:
      1. **Title**: Quickly grasp the topic.
      2. **Abstract**: This is the essence of the paper. Read it carefully to find answers to:
         1. Context: What is the problem domain?
         2. Problem: What specific pain point does it address?
         3. Method: What core method does it propose?
         4. Results: How good are the results?
      3. **Figures & Tables**: **This is the soul of a CV paper.**
         1. Prioritize the Architecture Figure: It visualizes the model's structure more intuitively than text.
         2. Look at the Results Table: Find the main results table, focusing on their method (usually the bolded row) and comparing it with the State-of-the-Art (SOTA) or key baselines. Did the metrics improve or decline?
         3. Examine Qualitative Results: Look at the generated images, detection boxes, or segmentation results to get an intuitive feel for the performance.
      4. **Introduction**: Skim through, focusing on the **last paragraph**. Authors usually summarize their contributions here.
      5. **Conclusion**: Read it quickly. It restates the core ideas and findings and points to future research directions.
    - After the first pass, you should be able to answer:
      - What problem does this paper solve?
      - What is its core idea?
      - What are its main achievements?
      - Is this paper worth more of my time?

  - **Second Pass: Dive into Details, Understand the Method (approx. 1 hour)**

    - Goal: Understand the specific implementation details of the method. **Focus on "how it's done" and "how it's evaluated."**
    - Reading Order and Content:
      1. **Methodology / Approach**: This is the core of this pass.
         1. Carefully re-read the architecture diagram and align it with the text description to ensure you understand how data flows through the model.
         2. Focus on Key Modules: Authors will typically detail their innovations (e.g., a new attention module, a new loss function). Understand the input, output, and internal logic of these modules.
         3. Mark Unfamiliar Terms and Formulas: Don't get stuck on complex mathematical derivations for now. Mark them and try to understand the **physical meaning** (the purpose) of the formulas first.
      2. **Experiments / Implementation Details**:
         1. Datasets: Which public datasets did they use? (e.g., ImageNet, COCO)
         2. Evaluation Metrics: What metrics did they use to measure performance? (e.g., mAP, IoU, PSNR)
         3. Baselines: Who did they mainly compare against?
         4. Training Details: Learning rate, optimizer, data augmentation, etc. This information is crucial for reproduction.
      3. **Ablation Studies**: **This is an extremely important part of CV papers!**
         1. Authors use a "control variables" approach to remove or replace their proposed new modules one by one to prove that **each innovation is effective**. Reading this part carefully will give you a deep understanding of the author's design philosophy and the contribution of each component.
    - After the second pass, you should be able to:
      - Clearly explain to others how the model works.
      - Understand how the authors proved their method's effectiveness through experiments.
      - Have a rough plan for how to reproduce the paper.

  - **Third Pass: Critical Thinking, "Virtual-Replication" (1-N hours)**

    - Goal: Achieve an expert-level understanding, be able to critically evaluate the work, and draw inspiration from it. **Focus on "why" and "what if."**
    - Reading Order and Content:
      1. Deep Dive into Mathematical Details: Go back to the formulas you marked in the second pass and try to derive them from scratch. Make sure you understand every assumption.
      2. Critically Examine the Method and Experiments:
         1. Ask "why": **Why** did the authors design it this way? Are there better alternatives?
         2. Look for "loopholes": Is the experimental comparison fair? Are there signs of "hyperparameter tuning"? Do the reported metrics hide performance degradation in some aspects? Are there strong baselines they deliberately avoided comparing against?
         3. Consider Limitations: Under what circumstances might this method fail?
      3. Read the Related Work Section in Detail:
         1. After understanding the paper's work, go back and read the related work section carefully. This will help you to more clearly position this paper in the broader knowledge map of the field and understand its **true innovation**.
      4. Find Points of Inspiration:
         1. Ask "what if": **What if** I apply this module to my own task? **What if** I combine the strengths of method A and method B?
         2. Pay attention to the **Future Work** mentioned in the conclusion, as these are often good research directions.
    - After the third pass, you should be able to:
      - Evaluate the paper's strengths and weaknesses comprehensively, like a reviewer.
      - Generate new research ideas or know how to apply it to your projects.
      - (Ideally) Be able to start reproducing the paper's core code.

  </details>

- Consistent Reading Goals:
  - What task does the paper accomplish: What is the input, what is the output? What are the difficulties of the task, and why do they exist?
  - What kind of pipeline does the paper design: What is the model architecture, and how is the loss function designed?
  - Where is the paper good, and why: What are the advantages of the designed architecture, and why does it have these advantages? How does it solve the difficulties?

### What Prompts to Use for AI-assisted Summary and Analysis?

- Tool Link: [alphaXiv](https://www.alphaxiv.org/)
- Model to Use: gemini-2.5-pro
- Simple Prompts:
  ```markdown
  1. What is the intuitive motivation for this research? How is it reflected in the technical design?
  2. Which existing work is most relevant to this study, and in what key aspects has it been improved or innovated upon?
  ```
- [Complex Prompt](./.prompts/cv_paper_notes_en.md):
  ```markdown
  # Prompts
  
  **Role Setting**: You are a seasoned researcher in computer graphics, adept at extracting deep insights from academic papers and organizing notes with high coherence and logical rigor. Your goal is to generate an in-depth research note that is not only comprehensive but also easy to understand and review.
  
  ## Core Processing Principles
  1.  **Strive for Coherence and Logic**: When generating notes, use fluent narrative language as much as possible, reducing unnecessary bullet points and isolated paragraphs. Ensure natural transitions and logical connections between different analysis modules (such as task description, methodology, experiments, advantage analysis, etc.) to form a cohesive, in-depth interpretation.
  2.  **Layered Reading and Information Integration**:
      *   **Quick Scan & Framework Building (10 minutes)**: Quickly read the abstract, introduction, and conclusion to understand the paper's core contributions and general structure.
      *   **Detailed Dive (Core Time)**: Carefully read key sections such as methodology, experimental design, network architecture, and loss functions.
      *   **Critical Thinking & Distillation**: Annotate innovations, potential limitations, and reflect on their underlying principles and potential impact.
  
  ## Note Structure and Specific Requirements
  Please organize the notes around the following structure, ensuring that each part is detailed and coherent:
  
  ---
  
  # Paper Title: [Paper Title] - [Conference/Journal] [Year]
  
  ### I. Introduction and Core Problem
  
  *   Briefly describe the research background and importance of the paper.
  *   **What is the core task the paper aims to solve?**
       *   **Input**: Describe in detail the type and specific form of the input data. For example, is it a single/multiple images, point clouds, text descriptions, or something else? Please specify its **data dimension/shape** (e.g., `[Batch_size, Channels, Height, Width]` for images, `[Batch_size, Num_points, Feature_dim]` for point clouds). If the input design is special (e.g., specific encoding methods, multimodal input fusion), please explain it systematically.
       *   **Output**: Describe in detail the type and specific form of the output data, also specifying its **data dimension/shape**. For example, is it a generated image, 3D model, segmentation mask, transformation parameters, or something else?
       *   **Task Application Scenarios**: List typical application scenarios of this task in computer graphics or other related fields.
       *   **Current Task Challenges (Pain Points)**: Deeply analyze the main difficulties faced by the current task. Why do these become difficulties (e.g., high computational complexity, data sparsity, lack of realism, weak generalization ability, etc.)?
       *   **Pain Points Addressed by the Paper**: Clearly state which of the above difficulties this paper focuses on designing and improving.
  
  ### II. Core Idea and Main Contributions
  
  *   **Intuitive Motivation and Design Embodiment**: What is the intuitive motivation for this research? How is this motivation reflected in the paper's technical design?
  *   **Comparison with Related Work and Innovation**: Which existing work(s) is this research most related to? In what key aspects does it improve upon or propose new ideas compared to these related works?
  *   **Core Contributions and Innovations**: Clearly distill the 1-3 most significant core contributions and innovations of the paper.
  
  ### III. Paper's Methodology (The Proposed Pipeline)
  
     *   **Overall Architecture Overview**: First, provide a coherent paragraph describing the overall pipeline of the proposed method/model.
     *   **Detailed Network Architecture and Data Flow**:
          *   Starting from **data preprocessing**, progressively describe how data flows through the network to the final output.
          *   **Layer-by-Layer/Module-by-Module Analysis**: Describe the design of each key layer or module in the network in detail. This includes, but is not limited to:
              *   **Layer/Module Type**: (e.g., Convolutional Layer, Transformer Encoder, Attention Mechanism, Custom-designed Block, etc.).
              *   **Design Details**: The specific configuration of the layer/module (e.g., kernel size, stride, channel changes, activation functions, normalization methods). If a module is novel or crucial, explain its internal structure and working principle in detail.
              *   **Shape Transformation**: Clearly explain the **shape or dimension change** of the data after passing through each layer/module.
              *   **Intermediate Variables**: Describe important intermediate feature representations and their significance.
              *   **Analysis of Role with Ablation Studies**: If the paper includes ablation studies, combine the experimental results to explain the specific contribution and role of each key layer/module or data processing step to the final performance.
     *   **Loss Function**:
          *   **Design Philosophy**: Describe the composition of the loss function in detail. What parts does it consist of? What is the mathematical form of each part?
          *   **Focus**: What aspects of information or constraints does the loss function primarily focus on (e.g., pixel-level reconstruction, perceptual similarity, structural consistency, adversarial loss, etc.)?
          *   **Training Implementation**: How is this loss function applied during the training process? Are there special weighting or scheduling strategies?
          *   **Contribution to Performance**: Is there evidence or analysis in the paper demonstrating the specific contribution of this loss function design to the final performance?
     *   **Dataset**:
          *   **Datasets Used**: Clearly state which public or private datasets were used for training and evaluation.
          *   **Special Handling**: Did the paper perform any special preprocessing, data augmentation, filtering, or construction on the datasets? If so, describe the specific methods and their stated purpose or effect.
  
  ### IV. Experimental Results and Analysis
  
     *   **Core Experimental Results**: Briefly summarize the main experimental results, interpreting them in coherent language. Key comparison tables from the paper can be used for assistance, but the focus is on interpretation. Please strictly use the data from the paper and do not fabricate data.
  
          | Metric     | Baseline A | Baseline B | Proposed Method | Improvement |
          |------------|------------|------------|-----------------|-------------|
          | [Metric 1] | ...        | ...        | ...             | ...         |
          | [Metric 2] | ...        | ...        | ...             | ...         |
  
     *   **Ablation Study Interpretation**: If the paper includes ablation studies, interpret their results in detail, clarifying the necessity and contribution of each component of the model.
     *   **Analysis of Visualization Results**: If available, describe and analyze representative visualization results from the paper and how they demonstrate the effectiveness of the method.
  
  ### V. Method Advantages and In-depth Analysis
  
     *   **Architecture/Design Advantages**:
          *   **Detailed Advantages**: Based on the previously described network architecture, loss function, and other implementation details, deeply analyze why the proposed method has advantages. For example, is it due to more efficient feature extraction, a more robust optimization target, better adaptation to specific data patterns, or other reasons?
          *   **Principle Elucidation**: Explain why these designs lead to such advantages. What are the underlying principles?
     *   **Idea and Practice for Solving Pain Points**: Summarize the core idea through which the paper addresses its targeted pain points and how it effectively solves them in practice through specific model design, training strategies, etc.
  
  ### VI. Conclusion and Personal Thoughts (Optional, but recommended)
  
     *   Review of the paper's main conclusions.
     *   **Potential Limitations**: Based on your understanding, point out potential limitations or unresolved issues of the method.
     *   **Future Work Directions**: Propose possible improvement directions or future research ideas based on this work.
     *   **Inspiration for Personal Research**: (If applicable) What inspiration does this paper offer for your own research?
  
  ### VII. Code Reference and Analysis Suggestions (If GitHub repository is accessible)
  
     *   **Repository Link**: [Provide if found]
     *   **Discussion of Core Module Implementation**: Based on the paper's description, suggest which core modules in the codebase to focus on (e.g., novel network layers, key algorithm flows). If time permits and the code is readable, briefly discuss the consistency of the code implementation with the paper's description or key implementation techniques. If not conducting a detailed code analysis, suggest that the reader consult and focus on specific modules, for example: "Readers are advised to consult the author's provided code [link], focusing on the implementation of [Module A] and [Module B] to understand their specific working methods and parameter configurations."
  
  ---
  
  **Handling of Technical Terms**:
  *   For technical terms in the field of computer graphics (e.g., BRDF, Monte Carlo Path Tracing, SDF, NeRF, Voxelization, Mesh Processing, etc.), provide a brief and clear explanation or annotation upon their first appearance.
  *   Use LaTeX format for mathematical formulas and symbols whenever possible (do not use "`"), and annotate the meaning of key variables.
  
  **Please strictly follow the above structure and requirements to conduct an in-depth interpretation of this paper and generate a coherent, detailed, and insightful research note.**
  
  Finally, please output the note in .md file format and display it within a code block. 
  ```

- Flowchart Drawing:

  ```markdown
  Please use mermaid to draw the flowchart of the *network architecture and data flow* in the paper.
  ```

## Table of Contents

- **Base/**
  - [202005_RAFT](Base/202005_RAFT.md)
  - **Diffusion_Models/**
    - [201511_DMs](Base/Diffusion_Models/201511_DMs.md)
    - [202012_DDPMs](Base/Diffusion_Models/202012_DDPMs.md)
    - [202106_ADMs](Base/Diffusion_Models/202106_ADMs.md)
    - [202201_EDMs](Base/Diffusion_Models/202201_EDMs.md)
    - [202204_LDMs](Base/Diffusion_Models/202204_LDMs.md)
    - [202305_DiT](Base/Diffusion_Models/202305_DiT.md)
    - [202312_CrossDiT](Base/Diffusion_Models/202312_CrossDiT.md)
    - [202312_UViT](Base/Diffusion_Models/202312_UViT.md)
    - [202405_MMDiT](Base/Diffusion_Models/202405_MMDiT.md)
    - [202501_FLUX_Context](Base/Diffusion_Models/202501_FLUX_Context.md)
    - [survey](Base/Diffusion_Models/survey.md)
  - **SSM/**
  - **Tokenization/**
    - [202201_iBOT](Base/Tokenization/202201_iBOT.md)
    - [202402_DINOv2](Base/Tokenization/202402_DINOv2.md)
    - [survey](Base/Tokenization/survey.md)
  - **Transformer/**
- **Generation/**
  - [202412_Imagen3](Generation/202412_Imagen3.md)
- **Human_Motion/**
  - [202208_QuickPose](Human_Motion/202208_QuickPose.md)
  - [202306_HMR3D](Human_Motion/202306_HMR3D.md)
  - [202308_HMR2.0](Human_Motion/202308_HMR2.0.md)
  - [202312_ViTPose++](Human_Motion/202312_ViTPose++.md)
  - [202403_TRAM](Human_Motion/202403_TRAM.md)
  - [202409_GVHMR](Human_Motion/202409_GVHMR.md)
  - [202501_JOSH](Human_Motion/202501_JOSH.md)
- **Reconsturction/**
  - [202503_VGGT](Reconsturction/202503_VGGT.md)
  - **CAD/**
    - [202308_PlainDETR](Reconsturction/CAD/202308_PlainDETR.md)
    - [202412_CubifyAnything](Reconsturction/CAD/202412_CubifyAnything.md)
    - [202503_LiteReality](Reconsturction/CAD/202503_LiteReality.md)
    - [202505_CAST](Reconsturction/CAD/202505_CAST.md)
    - **Ref/**
      - [202405_BlockFusion](Reconsturction/CAD/Ref/202405_BlockFusion.md)
      - [202411_BrepGen](Reconsturction/CAD/Ref/202411_BrepGen.md)
  - **Depth_Estimation/**
    - [202105_DPT](Reconsturction/Depth_Estimation/202105_DPT.md)
    - [202410_DepthAnythingV2](Reconsturction/Depth_Estimation/202410_DepthAnythingV2.md)
  - **Matching/**
    - [202305_RoMa](Reconsturction/Matching/202305_RoMa.md)
