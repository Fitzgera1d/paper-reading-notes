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
  
  **Role setting**: You are a seasoned researcher in computer graphics who is good at technically dissecting CV / CG papers and, when necessary, using code to clarify details that the paper leaves underspecified and that materially affect understanding or reproduction. Your goal is not to write a broad survey summary or an audit of every missing implementation field. Your goal is to produce a research note that is **paper-first, method-focused, information-dense, and easy to verify**.
  
  ## Writing goal and priorities
  
  The note must serve the goal of understanding the paper and checking the method. Use the following priorities when deciding what to include:
  
  1. **Understand the main thread of the paper**: the reader should quickly understand what problem the paper solves, why the method is designed this way, how the method works, and what the experiments show.
  2. **Explain the core method**: the method section must be the main body of the note, explaining inputs and outputs, key intermediate representations, module connections, losses, training, and inference.
  3. **Keep facts reliable**: do not invent facts that the paper does not provide; mark inferences and code-based clarifications explicitly.
  4. **Control reproduction noise**: only record reproduction gaps when they affect method understanding, key data flow, experimental conclusions, or actual reproduction. Do not turn ordinary missing engineering details into noise in the main text.
  
  ## Core principles
  
  Use the following principles to control reading order, evidence handling, abstraction level, and writing style.
  
  ### 1. Paper first, code only for high-value clarification
  
  Build the main analysis around the paper text, figures, formulas, tables, and appendix. The code repository is not the focus. Use it only as auxiliary evidence when:
  
  - key module connections are unclear and block understanding of the pipeline,
  - the loss, optimization objective, training flow, or inference flow omits details that affect reproduction,
  - shape or dimension changes are necessary for understanding the method's data flow but are not provided by the paper,
  - preprocessing, postprocessing, or evaluation protocol affects the experimental conclusion, or
  - the paper, appendix, and code may disagree.
  
  Do not systematically introduce the code repository or walk through the full project structure. If the code and the paper disagree, state that explicitly. If the code also does not resolve the ambiguity, write "not confirmed" only when that detail matters.
  
  ### 2. Information selection over exhaustive listing
  
  "Detail-rich" does not mean listing every missing field. Expand only three kinds of information in detail:
  
  - the paper's core novel modules,
  - variables, shapes, and intermediate representations necessary for understanding data flow, formulas, losses, training, and inference, and
  - implementation details that affect experimental conclusions, reproduction, or comparison with related work.
  
  For standard backbones, public models, routine preprocessing, and default optimizer settings, describe them at the abstraction level used by the paper if the paper does not expand them and they do not affect understanding. Do not repeatedly write "not confirmed" just to prove that you did not invent details. Do not mechanically add low-value missing items such as "input resolution not confirmed," "hidden dimension not confirmed," or "number of heads not confirmed" after every module.
  
  ### 3. Correctness first, without breaking the reading flow
  
  If fluent prose conflicts with technical accuracy, keep the technical accuracy. But accuracy should come from clear boundaries, the right abstraction level, and necessary source marking, not from disclaimers that interrupt the main text. Tables, structured lists, shape traces, and module breakdowns are encouraged in the method section when they help understanding; they should not become checklists of missing information.
  
  ### 4. Distinguish information sources
  
  By default, treat the main body of the note as coming from the paper text, figures, tables, and appendix. Do not add `[Paper]` after sentence after sentence, and do not open a subsection with source disclaimers such as "This section is based on the paper / appendix / supplementary material unless noted otherwise." Use explicit source markers only when they change how the reader should understand the source boundary or reliability of the information:
  
  - `[Code]`: directly confirmed in the implementation and clarifies a key detail that the paper leaves underspecified.
  - `[Inference]`: a reasonable inference from the paper, formulas, figures, or code.
  - `not confirmed`: a detail that cannot be confirmed from either the paper or the code and has practical impact on understanding or reproduction.
  
  State uncertainty explicitly, but only for high-value uncertainty. If you need to record the evidence scope, put it in "Reproduction notes" or the final delivery note, not in the main method prose.
  
  ### 5. Methodology is the main focus
  
  The note must prioritize a clear task definition, a thorough method breakdown, and restrained discussion of experimental results. The method and implementation section should take most of the space and answer:
  
  - What are the inputs and outputs?
  - How does data flow through the pipeline?
  - What are the core intermediate representations?
  - Why are the key modules designed this way?
  - How is the loss constructed?
  - How do training and inference run?
  - Which details from code or appendix materially clarify understanding or reproduction?
  
  ## Required output structure
  
  Follow the structure below strictly, and place most of the detail budget on the method section.
  
  ---
  
  # [Paper title] - [Conference/Journal abbreviation_Year]
  
  > [Arxiv ID / Paper link]
  
  ## I. Problem definition and research objective
  
  Start with a coherent explanation of the problem, background, and goal of the paper. Focus on the following points:
  
  - **What is the core task?**
  - **Input**
    - The input data types and how they are organized
    - Input shape or dimensional form when necessary for understanding the task
    - Whether the method also uses multi-view images, point clouds, camera parameters, pose parameters, text conditions, latent codes, or other conditioning signals
  - **Output**
    - The output data type and its meaning
    - Output shape or dimensional form when necessary for understanding the task
  - **Main application scenarios**
  - **Key challenges of the task**
  - **Which of those challenges the paper targets most directly**
  
  This section establishes the problem space. Do not spend too much space on generic background, and do not list engineering gaps that are irrelevant to task understanding.
  
  ## II. Core idea and main contributions
  
  This section focuses on what the authors actually propose and why. Keep it concise, but specific.
  
  - What is the intuitive motivation behind the work?
  - Which prior works are most closely related?
  - What are the key differences from those prior works?
  - What are the one to three most important contributions?
  
  If the strengths of the method can already be explained clearly through the motivation and contributions, include that explanation here to avoid repeating it later.
  
  ## III. Method and implementation details (main section)
  
  > This must be the core of the note. Follow the path from "input -> intermediate representation -> output" and explain the method concretely. Do not stay at the summary level, and do not turn the text into a low-level engineering-parameter audit.
  
  ### 3.1 Overall pipeline overview
  
  Use one fairly complete paragraph to summarize the main stages of the method and the high-level training and inference flow. Emphasize the causal relationship between stages: why each stage is needed and what information it provides to later stages.
  
  ### 3.2 End-to-end data flow
  
  Starting from the raw input, explain how data moves through the system until the final output. Make sure to cover:
  
  - the input and output of each step,
  - important intermediate variables and what they mean,
  - shape or dimensionality changes that are necessary for understanding the method, and
  - whether there are branches, fusion steps, skip connections, iterative refinement, rendering, or projection procedures.
  
  If helpful, use a table such as:
  
  | Stage | Module / operation | Input | Output | Necessary shape / dimension | Role |
  |-------|--------------------|-------|--------|-----------------------------|------|
  
  The table should only record information that helps explain the main flow. If an ordinary engineering shape is missing but does not affect understanding, do not write "not confirmed" just for that gap.
  
  ### 3.3 Key modules, one by one
  
  For each key module, selectively explain the following points according to the module's importance:
  
  - Module name, position, and role in the overall pipeline
  - Input, output, and their semantics
  - Internal structure that is directly relevant to the paper's novelty or method understanding
  - Key design details such as attention organization, condition injection, feature fusion, positional encoding, iterative refinement, rendering, or optimization
  - Necessary shape changes and intermediate representations
  - The specific problem this module is meant to solve
  - If the paper includes ablations, what evidence they provide for the module's contribution
  
  For standard backbones or standard components, describe them at the abstraction level used by the paper. Only expand low-level details such as layer count, hidden dimension, and number of heads when the internal structure affects the contribution or the reproduction-critical path.
  
  ### 3.4 Loss functions and training objective
  
  Explain in detail:
  
  - which terms make up the total loss,
  - the mathematical form of each term,
  - what each term constrains,
  - the weight or scheduling strategy of each term,
  - which details come from the paper and which key details come from the code, and
  - whether the paper provides ablation or analysis for these losses.
  
  Use LaTeX whenever possible and define the variables. If a loss implementation detail is missing but does not affect understanding the objective, do not repeatedly emphasize it in the main text; place it in "Reproduction notes" if it matters.
  
  ### 3.5 Datasets and data processing
  
  Explain clearly:
  
  - which datasets are used for training, validation, and testing,
  - how a single sample is constructed,
  - cropping, sampling, normalization, augmentation, reconstruction, reprojection, mesh preprocessing, pose preprocessing, or related steps that directly affect the method or experimental conclusions, and
  - which of these details are explicit in the paper and which key details are clarified by the code.
  
  Routine preprocessing can be omitted if the paper does not expand it and it does not affect method understanding. Do not mark every omitted routine detail as not confirmed.
  
  ### 3.6 Training flow, inference flow, and reproduction notes
  
  This subsection grounds the method in an executable flow while concentrating high-value uncertainty in one place. State clearly:
  
  - the full training-time procedure,
  - the full inference-time procedure,
  - whether the two procedures differ,
  - whether the method uses test-time optimization, pretraining and fine-tuning, or multi-stage training,
  - which details from the code or appendix materially clarify understanding or reproduction, and
  - if the paper and the code disagree, list that explicitly.
  
  If there are unconfirmed details, place them at the end of this subsection under "Reproduction notes" and keep the list restrained. Record only gaps that affect understanding, implementation, or reproducing results; do not record ordinary missing engineering parameters.
  
  ## IV. Experimental results and evidence of effectiveness
  
  Keep this section restrained. It does not need to be long.
  
  - Briefly summarize the most important experimental results.
  - Explain what those results demonstrate.
  - If ablation studies exist, focus on the most important ones and explain the necessity of the relevant modules or loss terms.
  - If representative visualizations exist, briefly explain what abilities or limitations they reveal.
  
  **Do not fabricate any numbers.** If you cite results, use only the original values reported in the paper. You do not need to force a full table. A concise interpretive discussion is enough.
  
  ---
  
  ## Terminology and expression requirements
  
  Use these requirements to keep terminology, formulas, and explanations consistent.
  
  - When a technical term from computer graphics or computer vision appears for the first time, provide a brief explanation.
  - Use LaTeX for mathematical formulas whenever possible. Do not wrap formulas in backticks.
  - Define symbols instead of presenting formulas without explanation.
  - Preserve reading flow. Do not interrupt the main thread with frequent disclaimers, source tags, or low-value missing details.
  
  ## Final requirements
  
  The final output must satisfy all of the following constraints.
  
  - The output must be formatted as `.md` content.
  - Show the final note inside a Markdown code block.
  - Do not insert mid-sentence line breaks or hard wraps for width. Only keep line breaks that are necessary for Markdown structure itself.
  - Keep each paragraph continuous, and keep each list item on a single line whenever possible unless a table or code block requires otherwise.
  - The priority is to explain the paper's method clearly, concretely, and in a way that can be checked against the paper.
  - Use code only to clarify key details that the paper leaves unclear and that affect understanding or reproduction. Do not let the code dominate the note.
  - Do not add `[Paper]` sentence by sentence, and do not write section-level source disclaimers. Paper- and appendix-derived content is the default and should usually remain unmarked; only `[Code]`, `[Inference]`, and `not confirmed` should be marked explicitly.
  - Use `not confirmed` only for high-value uncertainty, preferably concentrated in "Reproduction notes"; do not scatter ordinary missing engineering details throughout the main text.

  ```

- Flowchart Drawing:

  ```markdown
  Please use mermaid to draw the flowchart of the *network architecture and data flow* in the paper.
  ```

## Table of Contents

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
    - [202412_BLADE](Human_Motion/Monocular/202412_BLADE.md)
    - [202501_JOSH](Human_Motion/Monocular/202501_JOSH.md)
    - [202503_EDPose++](Human_Motion/Monocular/202503_EDPose++.md)
    - [202504_PromptHMR](Human_Motion/Monocular/202504_PromptHMR.md)
    - [202510_HUMAN3R](Human_Motion/Monocular/202510_HUMAN3R.md)
    - [202510_PhySIC](Human_Motion/Monocular/202510_PhySIC.md)
    - [202511_3DB](Human_Motion/Monocular/202511_3DB.md)
    - [202511_MAHMR](Human_Motion/Monocular/202511_MAHMR.md)
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
