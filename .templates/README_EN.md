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
- [Complex Prompt](./prompts/cv_paper_notes_en.md):
  ```markdown
  %content(./prompts/cv_paper_notes_en.md)
  ```

- Flowchart Drawing:

  ```markdown
  Please use mermaid to draw the flowchart of the *network architecture and data flow* in the paper.
  ```

## Table of Contents

%tree(./)
