# **扩散模型演进路线图：里程碑式进展解析**

**摘要**

本报告全面概述了扩散模型（一种杰出的生成式人工智能类别）的发展轨迹。报告追溯了其从基础理论概念到当前在各种领域最先进应用的发展历程。报告重点介绍了关键的里程碑式论文，详细阐述了它们的突破性创新、以引用量衡量的学术影响力，以及直接的arXiv链接，以供学术参考。本报告旨在提供一个结构化的视角，阐明这些模型如何克服了样本质量、计算效率和更广泛适用性方面的挑战，从而将其定位为人工智能领域的一项变革性技术。

## **1\. 引言：生成模型与扩散模型概述**

### **1.1. 生成建模领域概览**

生成建模是机器学习中的一个核心挑战，其目标是学习复杂的数据分布，从而生成新的、类似的数据样本。在历史上，这一领域主要由生成对抗网络（GANs）和变分自编码器（VAEs）等模型主导，每种模型都具有独特的优势，但也伴随着固有的局限性。GANs在生成高质量样本方面展现出卓越的能力，但它们经常面临训练不稳定和模式崩溃等问题，这限制了它们捕捉底层数据分布完整多样性的能力。相比之下，VAEs提供了更稳定的训练动态，但其生成的样本在视觉保真度上往往不如GANs。

GANs面临的持续挑战（即训练不稳定性和模式崩溃）以及VAEs的次优保真度，在生成建模领域造成了显著的未满足需求。这些问题不仅是技术瓶颈，更是阻碍生成式AI更广泛应用的关键障碍。这种对同时具备高质量输出和稳定训练的新范式的需求，为后来扩散模型的崛起奠定了基础。正是由于这些长期存在且影响深远的问题，研究人员才得以积极探索替代性的生成方法，而任何能够有效缓解这些特定问题的新模型类别，都必然会获得极大的关注和采纳。这种“问题-解决方案”的动态是AI创新发展的根本驱动力，它解释了扩散模型一旦展现出其能力便迅速获得关注的原因。它们不仅仅是一种新模型，更是对长期困扰生成式AI领域、具有高影响力的挑战的有力回应。

### **1.2. 扩散模型核心概念**

扩散模型（DMs），也被称为基于扩散的生成模型或基于分数的生成模型，是一类潜在变量生成模型。它们的基本操作原理深受非平衡态统计物理学概念的启发并根植于此。

**前向扩散过程**：这个过程涉及在一系列离散步骤中系统且逐渐地向数据中添加噪声，通常是高斯噪声。这种迭代的噪声注入逐步将原始的、复杂的数据分布转化为一个更简单、更易处理的分布，例如标准高斯分布。这个前向过程是确定性的，通常被建模为一个马尔可夫链，这意味着任何给定步骤的状态仅取决于其紧邻的前一个状态。

**反向采样过程**：扩散模型的核心创新和生成能力在于它们能够学习并有效地逆转噪声添加过程。一个神经网络被精心训练，以逐步去噪数据，从纯随机噪声状态开始，并迭代地将其转化回连贯且有意义的数据样本。这个学习到的反向过程正是生成新数据点的关键所在。

扩散模型固有的优雅之处在于它们能够将复杂的、高维生成问题（即学习复杂的、高维数据分布）转化为一系列更简单、计算上更易处理的去噪任务。这种迭代去噪机制，从根本上植根于统计物理学原理，自然地促进了训练过程的稳定性并有助于生成高度多样化的样本。这种核心设计选择直接解决了并缓解了早期生成模型中观察到的主要弱点。通过将复杂的生成过程分解为许多小而易于管理的去噪步骤，扩散模型在结构上与试图通过单一、通常不稳定的对抗性训练目标一次性学习整个数据分布的GANs形成了鲜明对比。这种“逐步”的扩散过程使得模型能够学习从简单噪声分布到复杂数据流形的整个连续路径，而不仅仅是近似最终数据分布。这种对数据空间的全面学习自然地促进了更好的模式覆盖和样本多样性。因此，这种受热力学原理启发的、多步骤、噪声逆转的基本设计选择，直接带来了训练稳定性的提高和生成样本多样性的增强，这代表了扩散模型成功的关键架构和理论突破。

## **2\. 基础奠定时期 (2015-2019)：理论基石**

### **2.1. 《使用非平衡态热力学进行深度无监督学习》(Deep Unsupervised Learning Using Nonequilibrium Thermodynamics) (Sohl-Dickstein et al., 2015\)**

* **论文标题**：Deep Unsupervised Learning Using Nonequilibrium Thermodynamics  
* **作者**：Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli  
* **年份**：2015  
* **arXiv 链接**：[arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
* **引用量**：8,105

**突破性创新**：这篇开创性论文将扩散模型作为一种基于热力学过程原理的生成框架引入。它详细阐述了迭代的前向过程（逐步添加噪声）和反向过程（学习去噪），从而奠定了扩散模型的数学和理论基础。作者们严谨地将概率模型定义为一个生成马尔可夫链的终点，该链系统地将一个简单的已知分布（例如高斯分布）转化为复杂的目标数据分布。此外，他们还提供了算法的开源参考实现，促进了早期采用和进一步研究。

**背景重要性**：在这项工作之前，生成模型面临着在实现高灵活性（建模复杂数据分布的能力）和保持可处理性（学习、采样和评估的便利性）之间的根本权衡。Sohl-Dickstein 等人巧妙地通过将前向扩散过程限制在一个简单、可解析的函数形式来解决这一难题。这种设计选择确保了相应的反向生成过程也保持可处理形式，使得整个框架既强大又实用。

Sohl-Dickstein 等人采用的“热力学过程”类比远不止是概念上的启发；它提供了一个严谨的数学框架，利用了科尔莫戈罗夫前向（福克-普朗克）和后向方程等既定概念。这种数学基础确保了扩散过程固有的可处理性和理论严谨性，这与当时其他生成模型往往缺乏此类深层理论基础形成了关键区别。该论文的标题和摘要明确指出“非平衡态统计物理学”和“热力学过程”是其灵感来源。更深入的探讨表明，文中提及了“科尔莫戈罗夫前向方程对应于福克-普朗克方程”以及“科尔莫戈罗夫后向方程描述了扩散过程的时间反转”。这表明模型行为是直接源自并受制于已被充分理解的物理定律及其数学表示。这种严谨的理论基础提供了“深刻的理论见解”，对于“长期进展和开发更高效、更有效算法”至关重要。它允许进行系统分析和改进，而不是仅仅依赖于经验性的试错，从而为扩散模型固有的稳定性及其最终的广泛应用做出了重大贡献。

## **3\. 现代扩散模型框架 (2020-2021)：结构定义与竞争力确立**

### **3.1. 《去噪扩散概率模型》(Denoising Diffusion Probabilistic Models, DDPM) (Ho et al., 2020\)**

* **论文标题**：Denoising Diffusion Probabilistic Models  
* **作者**：Jonathan Ho, Ajay N. Jain, Pieter Abbeel  
* **年份**：2020  
* **arXiv 链接**：[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
* **引用量**：21,717

**突破性创新**：DDPM 建立了现代的、实用的框架，将扩散模型推向了生成式 AI 的前沿。它展示了其生成高质量图像的卓越能力。该模型定义了一个用于前向噪声过程的马尔可夫扩散步链，以及一个用于数据生成的学习到的反向过程。至关重要的是，DDPM 发现，一个异常简化的训练目标——对噪声预测的基本均方误差损失——在实践中表现出惊人的效果，通常比直接优化复杂的变分下界能带来更优的样本质量。这一实用发现使得扩散模型在感知质量方面与 GANs 极具竞争力，同时提供了显著更稳定的训练和生成多样化样本的能力。

DDPM 凭借其出人意料的简化损失函数（均方误差噪声预测）实现了高质量图像生成的经验性展示，这标志着扩散模型发展的一个关键时刻。这种训练目标上的务实转变，摆脱了复杂的变分边界，使扩散模型在高质量图像生成方面变得更易于实现、更稳定且更具可扩展性。这一突破促进了其后续的快速扩展，证明了在不牺牲理论严谨性的前提下，可以实现实际效用。该论文的发现表明，DDPM 实现了“高质量图像合成”，并且在性能上“与 GANs 相当”。这得益于其采用了“简化目标”，特别是对噪声预测的均方误差损失。早期理论工作（例如 Sohl-Dickstein 等人的研究）涉及更复杂的变分下界，这在实践中优化起来可能具有挑战性。发现一个更简单的目标（Lsimple）在实践中表现更好，直接解决了“对抗性训练的优化挑战”，并使 DDPMs “更容易扩展和训练”。这种实用性上的简化对于释放扩散模型超越理论好奇心的全部潜力至关重要，使其在实际应用中变得可行。这体现了机器学习研究中一个经典的例子：一个理论上合理但可能复杂的模型，通过对其训练目标的经验性简化，变得在实践中可行并被广泛采用。它强调了理论严谨性与工程实用主义在推动 AI 进步中的关键相互作用，表明有时简化目标函数可以带来更大的实际影响并加速领域发展。

### **3.2. 《去噪扩散隐式模型》(Denoising Diffusion Implicit Models, DDIM) (Song et al., 2020\)**

* **论文标题**：Denoising Diffusion Implicit Models  
* **作者**：Jiaming Song, Chenlin Meng, Stefano Ermon  
* **年份**：2020  
* **arXiv 链接**：[https://arxiv.org/pdf/2010.02502](https://arxiv.org/pdf/2010.02502)
* **引用量**：6,629

**突破性创新**：DDIM 通过显著加速采样速度，对扩散模型的实用性带来了关键改进。这通过引入非马尔可夫扩散过程实现。与原始 DDPM 需要模拟大量步骤的马尔可夫链不同，DDIM 实现了更快的采样速度——快了 10 到 50 倍——同时精确地保持了高样本质量。此外，这项创新促进了在潜在空间中直接进行语义上有意义的图像插值。这一进展的核心在于构建了一类非马尔可夫扩散过程，尽管它们是非马尔可夫性质的，但却产生了与 DDPM 相同的训练目标，并允许使用“短”生成马尔可夫链进行高效采样。

**背景重要性**：早期 DDPM 的一个显著限制是其缓慢的采样速度，通常需要数千个迭代步骤才能生成单个图像。这使得它们在计算上过于昂贵，并且比 GANs 慢得多。DDIM 直接解决了这一效率瓶颈，从而使扩散模型在实际应用中更具实用性和可部署性。

DDIM 在采样效率方面的突破对于扩散模型的实际可行性和广泛采用至关重要。如果没有这些显著的速度提升，它们在其他方面卓越的样本质量将因过高的计算成本和缓慢的推理时间而主要局限于学术研究。这项创新标志着该领域的一个关键转变，将重点从仅仅实现高质量转向确保实际可部署性和现实世界效用。DDPM 在生成高质量图像方面表现出色，但其采样速度“比 GANs 慢得多”，并且“生成一个样本需要数千步”。这种缓慢的推理速度是任何生成模型实际应用和商业可行性的主要障碍。即使高质量的结果，如果生成时间过长，其影响力也会大打折扣。DDIM 通过引入一种“更快采样，减少所需步骤”的方法，并实现了“壁钟时间快 10 到 50 倍”的速度，直接解决了这个关键瓶颈。这不仅仅是渐进式的改进，而是对采样过程的根本性重新设计。这突出了 AI 发展中一个反复出现且至关重要的主题：模型能力的初步突破之后，通常需要进行重大的工程和算法优化，才能实现实际效用和广泛采用。DDIM 有效地使扩散模型在更广泛的应用范围内变得“可用”，加速了它们在各个行业的整合。

### **3.3. 《改进的去噪扩散概率模型》(Improved Denoising Diffusion Probabilistic Models) (Nichol & Dhariwal, 2021\)**

* **论文标题**：Improved Denoising Diffusion Probabilistic Models  
* **作者**：Alexander Quinn Nichol, Prafulla Dhariwal  
* **年份**：2021  
* **arXiv 链接**：[http://arxiv.org/pdf/2102.09672](http://arxiv.org/pdf/2102.09672)
* **引用量**：4,206

**突破性创新**：这篇论文通过几项关键修改显著改进了 DDPM，其中最显著的是引入了改进的噪声调度和开创性的无分类器引导概念。他们证明，通过学习反向扩散过程的方差，可以在显著减少前向传递次数（例如，仅需 50 步，而原始 DDPM 需要数百步）的情况下实现采样，同时样本质量损失可忽略不计。此外，这项工作证明 DDPM 即使在 ImageNet 等高度多样化的数据集上也能获得具有竞争力的对数似然。

**背景重要性**：这项研究在 DDPM 成功的基础上，进一步完善了核心扩散模型，以提高图像质量和采样效率，使其在实际部署中更加稳健和实用。无分类器引导的引入成为一种标准且极具影响力的技术，用于有效管理条件生成任务中多样性和保真度之间的权衡。

无分类器引导的引入是条件扩散模型领域的一项范式转变创新。它提供了一种极其简单但功能强大的机制，可以精确控制生成样本的多样性与其对给定条件输入（例如文本提示）的依从性之间的权衡。这一能力极大地提高了生成内容的实用性、可控性和艺术表现力，使扩散模型超越了单纯的无条件生成，成为高度可控的创作工具。该论文的关键贡献包括“无分类器引导”。这种引导方法通过利用分类器的梯度，实现了“多样性与保真度的权衡”。这意味着对生成过程具有直接、可调的影响。这种明确的控制机制至关重要，因为它允许用户或下游系统根据具体需求微调输出。例如，在文本到图像生成中，人们可能优先考虑严格遵循文本提示（保真度）而不是广泛的创意变化（多样性），反之亦然。这种程度的控制在以前的模型中并不容易获得或直观。这种创新将扩散模型从强大的无条件生成模型（生成没有特定指令的图像）转变为高度可控的条件生成模型（根据文本、类别标签或其他输入生成图像）。这种功能扩展显著拓宽了其在实际应用中的适用性，特别是对于文本到图像合成等任务，这后来成为一项重要的商业成功。

### **3.4. 《扩散模型在图像合成方面超越GANs》(Diffusion Models Beat GANs on Image Synthesis) (Dhariwal & Nichol, 2021\)**

* **论文标题**：Diffusion Models Beat GANs on Image Synthesis  
* **作者**：Prafulla Dhariwal, Alexander Nichol  
* **年份**：2021  
* **arXiv 链接**：[https://ar5iv.labs.arxiv.org/html/2105.05233](https://ar5iv.labs.arxiv.arxiv.org/html/2105.05233) 
* **引用量**：8,805

**突破性创新**：这篇里程碑式的论文提供了确凿的经验证据，表明扩散模型，通过架构改进和分类器引导的策略性应用，可以在 ImageNet 等极具挑战性的数据集上生成超越 GANs 的高质量图像。它们实现了新的最先进的 Fréchet Inception Distance (FID) 分数，甚至在某些情况下以更少的前向传递次数，匹配或超越了 BigGAN-deep 等领先 GAN 架构的性能。这项关键工作巩固了扩散模型作为无条件图像合成领域新一代最先进范式的地位。

**背景重要性**：多年来，GANs 被广泛认为是实现高保真图像生成的黄金标准。这篇论文提供了无可辩驳的经验证据，表明扩散模型不仅能够匹敌，甚至超越 GAN 的性能，从而从根本上改变了主流研究范式，并将生成式 AI 社区的焦点转向了扩散模型。

这篇论文代表了扩散模型发展的转折点。通过明确证明其在经验上优于 GANs——长期以来的黄金标准——它引发了全球范围内研究、开发和投资的大规模激增。这个关键时刻有效地将整个生成式 AI 社区的焦点和资源转向了扩散模型，将其确立为高质量生成任务的主导范式。这不仅仅是渐进式的改进，而是一次根本性的“去王冠化”，重塑了生成式 AI 的格局。该论文的标题本身就提出了一个大胆的主张：“扩散模型在图像合成方面超越 GANs”。这不是一个细微的发现；这是对既定最先进技术的直接挑战。在此之前，GANs 被明确认为是“最先进的”，并且是高保真图像合成领域的主导生成模型。该论文中提出的经验结果，特别是 ImageNet 等挑战性数据集上优越的 FID 分数，提供了令人信服且无可否认的证据，表明扩散模型可以超越 GANs。这种直接的经验验证对于说服更广泛的 AI 社区放弃 GANs 并大力投资于扩散模型的研究和开发至关重要。它提供了许多研究人员所需要的“概念验证”。这清晰地展示了重大经验突破如何从根本上重塑整个研究领域。该论文的发现导致研究工作、资金和人才迅速重新分配到扩散模型上，以前所未有的速度加速了其在各个领域的发展和应用。它标志着 GANs 无可争议的统治的结束，以及生成式 AI“扩散时代”的开始。

## **4\. 进阶架构与效率 (2022-至今)：扩展与优化**

### **4.1. 《使用潜在扩散模型进行高分辨率图像合成》(High-Resolution Image Synthesis with Latent Diffusion Models) (Rombach et al., 2022\)**

* **论文标题**：High-Resolution Image Synthesis with Latent Diffusion Models  
* **作者**：Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
* **年份**：2022  
* **arXiv 链接**：[arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
* **引用量**：18,947

**突破性创新**：这篇论文引入了潜在扩散模型（LDMs），这是一种变革性的方法，极大地降低了高分辨率图像合成的计算成本。这是通过在压缩的、低维潜在空间中执行整个扩散过程来实现的，而不是在计算开销大的高维像素空间中进行。这一创新框架成功地平衡了视觉质量和计算效率，使得高分辨率合成对更广泛的用户和应用来说更易于实现。此外，LDMs 集成了交叉注意力机制，实现了灵活而强大的文本条件图像合成，这是现代生成应用的关键特性。这项开创性工作为广泛采用的商业和开源模型（最著名的是 Stable Diffusion）奠定了基础。

**背景重要性**：尽管之前的扩散模型在图像质量方面取得了令人印象深刻的成果，但在高分辨率下生成图像仍然计算成本高昂且资源密集。LDMs 通过将计算密集型扩散过程转移到低维潜在空间，直接解决了这一关键瓶颈。这一创新有效地普及了高分辨率图像生成，使其对更广泛的研究人员和从业者群体来说变得可行。

LDM 将扩散过程在压缩潜在空间中执行的创新，是扩散模型普及化和后续商业化的关键推动力。通过在不牺牲输出质量的前提下大幅降低高分辨率图像合成的计算需求，它将扩散模型从一种专业的科研好奇心转变为一种实用、广泛可用的工具。这直接导致了 Stable Diffusion 等用户友好型文本到图像应用的爆炸式增长，标志着从纯粹的科学突破到广泛的社会和工业影响的重大转变。LDMs 被明确认为通过在“潜在空间”中操作，降低了“计算成本”和“计算复杂性”。这是多个信息来源中一致的观点。之前，使用扩散模型生成高分辨率图像“在内存和时间上都很昂贵”，并且计算密集，限制了其广泛使用。通过降低计算门槛，LDMs 使得更多的研究人员、开发人员，甚至拥有消费级硬件的个人，能够更可行地训练和部署高分辨率生成模型。这种资源需求的直接降低是实现 Stable Diffusion 等广泛可用的模型开发和普及的关键因素，这些模型可以在更适中的硬件上运行。这清楚而有力地说明了效率提升如何从根本上解锁新的应用并促进广泛采用。它将扩散模型从一个主要的研究领域转变为一个具有深远商业、创意甚至社会影响的主流技术，催生了如今被数百万人使用的工具。

### **4.2. 《阐明基于扩散的生成模型的设计空间》(Elucidating the Design Space of Diffusion-Based Generative Models) (Karras et al., 2022\)**

* **论文标题**：Elucidating the Design Space of Diffusion-Based Generative Models  
* **作者**：Tero Karras, Miika Aittala, Timo Aila, Samuli Laine  
* **年份**：2022  
* **arXiv 链接**：[https://arxiv.org/abs/2206.00364](https://arxiv.org/abs/2206.00364)
* **引用量**：1,609

**突破性创新**：这篇论文对扩散模型中复杂的设计选择进行了开创性的、全面而系统的分析。它有效地简化了这些模型中常常令人困惑的理论和实践方面。作者们细致地识别并阐明了各种组件的最佳配置，包括噪声调度、网络架构和采样过程。这种系统方法带来了新的最先进的 FID 分数，同时实现了显著更快的采样率（例如，CIFAR-10 每张图像仅需 35 次网络评估）。实质上，这项工作“阐明”了构建和训练高效、有效扩散模型的最佳实践和基本原理。

**背景重要性**：随着扩散模型迅速变得复杂和流行，它们的设计空间变得越来越广阔，对研究人员和从业者来说常常令人困惑。Karras 等人的工作提供了急需的清晰度和系统路线图，用于未来的发展，从而实现更有针对性、更高效和更原则性的研究工作。

这篇论文构成了关键的元级贡献，从根本上将研究重点从仅仅发现和展示扩散模型，转向系统地优化和标准化其设计和实现。通过细致地描绘设计空间并识别最佳配置，它为构建更高效、更高性能的模型提供了科学蓝图，从而加速了该领域的整体成熟。这项工作体现了“AI 科学”如何深刻地改进“AI 实践”。该论文的摘要和内容反复强调其目标是“通过提供一个明确区分具体设计选择的设计空间来纠正现状”，并“阐明（这次是从理论上）如何设计训练和采样过程以实现有效生成”。在此工作之前，扩散模型文献“理论密集”但“不必要地复杂”，使得研究人员难以系统地改进模型。通过提供结构化分析、识别关键自由度，并提出各种组件（例如噪声调度、采样器、网络架构）的最佳配置，该论文为构建更好的模型提供了清晰的指导。这直接导致了“新的最先进 FID”和“更快的采样”，展示了系统设计的实际影响。这项工作代表了任何快速发展研究领域成熟的关键阶段。一旦新的模型类别出现并证明其潜力，下一步的关键就是对其底层组件进行系统理解和优化。这篇论文为扩散模型提供了正是这一关键步骤，将该领域从快速、通常是临时性的经验实验阶段，转变为更具原则性、工程驱动的开发阶段，确保了长期可持续的进展。

## **5\. 更广泛的应用与未来方向**

### **5.1. 应用多样化**

除了在图像生成方面的初步成功，扩散模型已迅速扩展到一系列令人印象深刻的多元应用中，这突显了其固有的多功能性和在不同数据模态中强大的生成能力。

* **视频生成**：诸如《视频扩散模型》(Video Diffusion Models) (Ho et al., 2022\) 等开创性工作成功地将扩散框架扩展到学习和合成时间动态，以实现逼真的视频生成。  
* **音频合成**：像 DiffWave (Kong et al., 2021\) 这样的模型展示了扩散模型在各种音频任务中的有效应用，包括文本到语音（TTS）和通用音频生成。  
* **自然语言处理 (NLP)**：扩散模型已在 NLP 任务中找到应用，包括文本生成和摘要，展示了它们处理序列数据的能力。  
* **医学影像**：这是一个特别有影响力的领域，扩散模型被用于合成图像生成、去噪、超分辨率、分割和数据增强，以解决临床数据集中数据稀缺和类别不平衡等挑战。  
* **其他领域**：扩散模型还扩展到分子设计、3D 场景渲染、强化学习、生物信息学和故障诊断等领域。

### **5.2. 持续挑战与未来展望**

尽管扩散模型取得了显著进展，但仍面临一些持续的挑战。对这些模型的理论理解，尽管经验上取得了巨大成功，但仍相对有限。此外，它们在计算资源方面的需求仍然很高，尤其是在处理高分辨率数据时。随着生成能力的增强，关于偏见、滥用和伦理影响的担忧也日益突出。

未来的研究方向将集中在：

* **提升效率**：进一步优化模型架构和采样策略，以降低计算成本并加速生成过程。  
* **增强可控性**：开发更精细的控制机制，以实现对生成内容更精确的引导和编辑。  
* **多模态生成**：探索更强大的模型，能够无缝地处理和生成多种数据模态（如文本、图像、音频和视频的组合）。  
* **新型应用**：将扩散模型应用于新的科学和工程领域，解决目前难以解决的问题。  
* **理论进展**：深化对扩散模型底层机制的理论理解，为更稳健和可解释的算法设计提供指导。

## **6\. 结论**

扩散模型自其在非平衡态热力学中找到理论根基以来，已迅速发展成为生成式人工智能领域的一股变革性力量。从 Sohl-Dickstein 等人奠定基础，到 Ho 等人的 DDPM 确立现代框架并证明其与 GANs 的竞争力，再到 Song 等人的 DDIM 显著提升采样效率，以及 Nichol 和 Dhariwal 的工作通过无分类器引导进一步精炼模型并最终超越 GANs，每一步都代表着关键的科学和工程突破。Rombach 等人引入的潜在扩散模型（LDMs）通过在潜在空间中操作，极大地降低了计算成本，从而普及了高分辨率图像生成，并为 Stable Diffusion 等广泛应用奠定了基础。Karras 等人对设计空间的系统阐明，则为该领域的持续优化和标准化提供了蓝图。

这些里程碑式进展不仅提升了生成样本的质量和多样性，还解决了训练稳定性、计算效率和可控性等关键瓶颈。扩散模型已从最初的图像合成扩展到视频、音频、自然语言处理和医学成像等多元领域，展现出其作为通用生成工具的强大潜力。尽管仍面临理论理解、计算需求和伦理考量等挑战，但扩散模型的持续演进预示着未来在人工智能领域将有更深远的创新和更广泛的应用。其发展轨迹清晰地描绘了一个从理论探索到实用突破，再到广泛应用和持续优化的典范，预示着其在塑造人工智能未来方面将扮演核心角色。

#### **引用的著作**

1. Deep Unsupervised Learning using Nonequilibrium Thermodynamics, 访问时间为 五月 21, 2025， [http://proceedings.mlr.press/v37/sohl-dickstein15.pdf](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf)  
2. Diffusion models in bioinformatics and computational biology \- PMC \- PubMed Central, 访问时间为 五月 21, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10994218/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10994218/)  
3. Diffusion model \- Wikipedia, 访问时间为 五月 21, 2025， [https://en.wikipedia.org/wiki/Diffusion\_model](https://en.wikipedia.org/wiki/Diffusion_model)  
4. Diffusion Models Beat GANs on Image Synthesis \- NIPS papers, 访问时间为 五月 21, 2025， [https://papers.nips.cc/paper\_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf](https://papers.nips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)  
5. Opportunities and challenges of diffusion models for generative AI \- Oxford Academic, 访问时间为 五月 21, 2025， [https://academic.oup.com/nsr/article/11/12/nwae348/7810289](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)  
6. Denoising Diffusion Probabilistic Models and Transfer Learning for citrus disease diagnosis, 访问时间为 五月 21, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10749533/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10749533/)  
7. denoising diffusion probabilistic model with wavelet packet transform for fingerprint generation \- JJCIT, 访问时间为 五月 21, 2025， [https://jjcit.org/paper/241](https://jjcit.org/paper/241)  
8. Denoising Diffusion Implicit Model Combined with TransNet for Rolling Bearing Fault Diagnosis Under Imbalanced Data \- MDPI, 访问时间为 五月 21, 2025， [https://www.mdpi.com/1424-8220/24/24/8009](https://www.mdpi.com/1424-8220/24/24/8009)  
9. \[1503.03585\] Deep Unsupervised Learning using Nonequilibrium Thermodynamics \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585)  
10. Denoising Diffusion Probabilistic Models, 访问时间为 五月 21, 2025， [https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)  
11. Improved Denoising Diffusion Probabilistic Models \- arXiv, 访问时间为 五月 21, 2025， [http://arxiv.org/pdf/2102.09672](http://arxiv.org/pdf/2102.09672)  
12. Novel Paintings from the Latent Diffusion Model through Transfer Learning \- MDPI, 访问时间为 五月 21, 2025， [https://www.mdpi.com/2076-3417/13/18/10379](https://www.mdpi.com/2076-3417/13/18/10379)  
13. Elucidating the Design Space of Diffusion-Based Generative Models, 访问时间为 五月 21, 2025， [https://users.aalto.fi/\~laines9/publications/karras2022elucidating\_paper.pdf](https://users.aalto.fi/~laines9/publications/karras2022elucidating_paper.pdf)  
14. Elucidating the Design Space of Diffusion-Based Generative Models \- OpenReview, 访问时间为 五月 21, 2025， [https://openreview.net/pdf?id=k7FuTOWMOc7](https://openreview.net/pdf?id=k7FuTOWMOc7)  
15. Denoising diffusion probabilistic models for generation of realistic fully-annotated microscopy image datasets \- PMC, 访问时间为 五月 21, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10906858/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10906858/)  
16. Representation learning with unconditional denoising diffusion models for dynamical systems \- EGUsphere, 访问时间为 五月 21, 2025， [https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2261/egusphere-2023-2261.pdf](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2261/egusphere-2023-2261.pdf)  
17. Denoising Diffusion Implicit Models \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/pdf/2010.02502](https://arxiv.org/pdf/2010.02502)  
18. From denoising diffusions to denoising Markov models | Journal of the Royal Statistical Society Series B \- Oxford Academic, 访问时间为 五月 21, 2025， [https://academic.oup.com/jrsssb/article/86/2/286/7564909](https://academic.oup.com/jrsssb/article/86/2/286/7564909)  
19. On Denoising Diffusion Probabilistic Models for Synthetic Aperture Radar Despeckling, 访问时间为 五月 21, 2025， [https://www.mdpi.com/1424-8220/25/7/2149](https://www.mdpi.com/1424-8220/25/7/2149)  
20. Denoising diffusion model for increased performance of detecting structural heart disease \- PMC, 访问时间为 五月 21, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC11601717/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601717/)  
21. Improved Outcome Models with Denoising Diffusion \- PMC, 访问时间为 五月 21, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10939775/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10939775/)  
22. denoising diffusion probabilistic model with wavelet packet transform for fingerprint generation \- JJCIT, 访问时间为 五月 21, 2025， [https://www.jjcit.org/paper/241](https://www.jjcit.org/paper/241)  
23. Diffusion Models in Recommendation Systems: A Survey \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/html/2501.10548v2](https://arxiv.org/html/2501.10548v2)  
24. Physics and data-driven alternative optimization enabled ultra-low-sampling single-pixel imaging \- SPIE Digital Library, 访问时间为 五月 21, 2025， [https://www.spiedigitallibrary.org/journals/advanced-photonics-nexus/volume-4/issue-03/036005/Physics-and-data-driven-alternative-optimization-enabled-ultra-low-sampling/10.1117/1.APN.4.3.036005.full](https://www.spiedigitallibrary.org/journals/advanced-photonics-nexus/volume-4/issue-03/036005/Physics-and-data-driven-alternative-optimization-enabled-ultra-low-sampling/10.1117/1.APN.4.3.036005.full)  
25. Latent Diffusion Models to Enhance the Performance of Visual Defect Segmentation Networks in Steel Surface Inspection \- MDPI, 访问时间为 五月 21, 2025， [https://www.mdpi.com/1424-8220/24/18/6016](https://www.mdpi.com/1424-8220/24/18/6016)  
26. Denoising Diffusion Probabilistic Models | Request PDF \- ResearchGate, 访问时间为 五月 21, 2025， [https://www.researchgate.net/publication/342352924\_Denoising\_Diffusion\_Probabilistic\_Models](https://www.researchgate.net/publication/342352924_Denoising_Diffusion_Probabilistic_Models)  
27. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/html/2502.07225v1](https://arxiv.org/html/2502.07225v1)  
28. Spread Spectrum Image Watermarking Through Latent Diffusion Model \- MDPI, 访问时间为 五月 21, 2025， [https://www.mdpi.com/1099-4300/27/4/428](https://www.mdpi.com/1099-4300/27/4/428)  
29. A chronological timeline of key research papers on diffusion models, including links to arXiv papers and summaries of their contributions. \- GitHub, 访问时间为 五月 21, 2025， [https://github.com/jeffreybarry/Diffusion-Research-Timeline](https://github.com/jeffreybarry/Diffusion-Research-Timeline)  
30. ‪Jascha Sohl-Dickstein‬ \- ‪Google Scholar‬, 访问时间为 五月 21, 2025， [https://scholar.google.com/citations?user=-3zYIjQAAAAJ\&hl=en](https://scholar.google.com/citations?user=-3zYIjQAAAAJ&hl=en)  
31. Diffusion Models Basic Principles \- UA Campus Repository, 访问时间为 五月 21, 2025， [https://repository.arizona.edu/handle/10150/675496](https://repository.arizona.edu/handle/10150/675496)  
32. hojonathanho/diffusion: Denoising Diffusion Probabilistic Models \- GitHub, 访问时间为 五月 21, 2025， [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)  
33. ‪Jonathan Ho‬ \- ‪Google Scholar‬, 访问时间为 五月 21, 2025， [https://scholar.google.com/citations?user=iVLAQysAAAAJ\&hl=en](https://scholar.google.com/citations?user=iVLAQysAAAAJ&hl=en)  
34. Diffusion Probabilistic Modeling for Video Generation \- MDPI, 访问时间为 五月 21, 2025， [https://www.mdpi.com/1099-4300/25/10/1469](https://www.mdpi.com/1099-4300/25/10/1469)  
35. Denoising Diffusion Probabilistic Models （DDPM), 访问时间为 五月 21, 2025， [https://www.cs.jhu.edu/\~ayuille/JHUcourses/VisionAsBayesianInference2025/22/Lecture22\_diffusion.pdf](https://www.cs.jhu.edu/~ayuille/JHUcourses/VisionAsBayesianInference2025/22/Lecture22_diffusion.pdf)  
36. Physics-Inspired Approaches in Generative Diffusion Models \- JPS Journals, 访问时间为 五月 21, 2025， [https://journals.jps.jp/doi/abs/10.7566/JPSJ.94.031008?mobileUi=0](https://journals.jps.jp/doi/abs/10.7566/JPSJ.94.031008?mobileUi=0)  
37. Frequency-Aware Diffusion Model for Multi-Modal MRI Image Synthesis \- MDPI, 访问时间为 五月 21, 2025， [https://www.mdpi.com/2313-433X/11/5/152](https://www.mdpi.com/2313-433X/11/5/152)  
38. Opportunities and challenges of diffusion models for generative AI \- Oxford Academic, 访问时间为 五月 21, 2025， [https://academic.oup.com/nsr/article/11/12/nwae348/7810289?login=false](https://academic.oup.com/nsr/article/11/12/nwae348/7810289?login=false)  
39. Comparative Analysis of GANs and Diffusion Models in Image Generation | Highlights in Science, Engineering and Technology \- Darcy & Roy Press, 访问时间为 五月 21, 2025， [https://drpress.org/ojs/index.php/HSET/article/view/28780](https://drpress.org/ojs/index.php/HSET/article/view/28780)  
40. ‪Prafulla Dhariwal‬ \- ‪Google Scholar‬, 访问时间为 五月 21, 2025， [https://scholar.google.com/citations?user=0pOgVVAAAAAJ\&hl=en](https://scholar.google.com/citations?user=0pOgVVAAAAAJ&hl=en)  
41. ‪Prafulla Dhariwal‬ \- ‪Google Scholar‬, 访问时间为 五月 21, 2025， [https://scholar.google.de/citations?user=0pOgVVAAAAAJ\&hl=th](https://scholar.google.de/citations?user=0pOgVVAAAAAJ&hl=th)  
42. \[2105.05233\] Diffusion Models Beat GANs on Image Synthesis \- ar5iv \- arXiv, 访问时间为 五月 21, 2025， [https://ar5iv.labs.arxiv.org/html/2105.05233](https://ar5iv.labs.arxiv.org/html/2105.05233)  
43. Diffusion Models Beat GANs on Image Synthesis \- NIPS papers, 访问时间为 五月 21, 2025， [https://papers.nips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html](https://papers.nips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)  
44. Diffusion Models Beat GANs on Image Synthesis \- ResearchGate, 访问时间为 五月 21, 2025， [https://www.researchgate.net/publication/351512977\_Diffusion\_Models\_Beat\_GANs\_on\_Image\_Synthesis](https://www.researchgate.net/publication/351512977_Diffusion_Models_Beat_GANs_on_Image_Synthesis)  
45. Tero Karras's research while affiliated with NVIDIA and other places \- ResearchGate, 访问时间为 五月 21, 2025， [https://www.researchgate.net/scientific-contributions/Tero-Karras-57150323](https://www.researchgate.net/scientific-contributions/Tero-Karras-57150323)  
46. ‪Robin Rombach‬ \- ‪Google Scholar‬, 访问时间为 五月 21, 2025， [https://scholar.google.com/citations?user=ygdQhrIAAAAJ\&hl=en](https://scholar.google.com/citations?user=ygdQhrIAAAAJ&hl=en)  
47. Stable diffusion for high-quality image reconstruction in digital rock analysis \- SciOpen, 访问时间为 五月 21, 2025， [https://www.sciopen.com/article/10.46690/ager.2024.06.02](https://www.sciopen.com/article/10.46690/ager.2024.06.02)  
48. Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/html/2503.18352v1](https://arxiv.org/html/2503.18352v1)  
49. Synthetic Diffusion Tensor Imaging Maps Generated by 2D and 3D Probabilistic Diffusion Models: Evaluation and Applications \- PMC, 访问时间为 五月 21, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC11888198/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888198/)  
50. Robin Rombach's research while affiliated with Ludwig-Maximilians-Universität in Munich and other places \- ResearchGate, 访问时间为 五月 21, 2025， [https://www.researchgate.net/scientific-contributions/Robin-Rombach-2174164171](https://www.researchgate.net/scientific-contributions/Robin-Rombach-2174164171)  
51. \[2503.18352\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/abs/2503.18352](https://arxiv.org/abs/2503.18352)  
52. NVlabs/edm: Elucidating the Design Space of Diffusion-Based Generative Models (EDM) \- GitHub, 访问时间为 五月 21, 2025， [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)  
53. Physics-Inspired Generative Models in Medical Imaging \- Annual Reviews, 访问时间为 五月 21, 2025， [https://www.annualreviews.org/content/journals/10.1146/annurev-bioeng-102723-013922](https://www.annualreviews.org/content/journals/10.1146/annurev-bioeng-102723-013922)  
54. Evaluating the design space of diffusion-based generative models \- ResearchGate, 访问时间为 五月 21, 2025， [https://www.researchgate.net/publication/381518421\_Evaluating\_the\_design\_space\_of\_diffusion-based\_generative\_models](https://www.researchgate.net/publication/381518421_Evaluating_the_design_space_of_diffusion-based_generative_models)  
55. Appendices: Elucidating the Design Space of Diffusion-Based Generative Models \- NIPS papers, 访问时间为 五月 21, 2025， [https://proceedings.neurips.cc/paper\_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Supplemental-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Supplemental-Conference.pdf)  
56. Tero Karras \- Research at NVIDIA, 访问时间为 五月 21, 2025， [https://research.nvidia.com/person/tero-karras](https://research.nvidia.com/person/tero-karras)  
57. CoCoGen: Physically Consistent and Conditioned Score-Based Generative Models for Forward and Inverse Problems | SIAM Journal on Scientific Computing, 访问时间为 五月 21, 2025， [https://epubs.siam.org/doi/abs/10.1137/24M1636071](https://epubs.siam.org/doi/abs/10.1137/24M1636071)  
58. Efficient Diffusion Models: A Comprehensive Survey from Principles to Practices \- arXiv, 访问时间为 五月 21, 2025， [https://arxiv.org/html/2410.11795v1](https://arxiv.org/html/2410.11795v1)  
59. Diffusion Models for Medical Image Computing: A Survey \- SciOpen, 访问时间为 五月 21, 2025， [https://www.sciopen.com/article/10.26599/TST.2024.9010047](https://www.sciopen.com/article/10.26599/TST.2024.9010047)  
60. A diffusion-based super resolution model for enhancing sonar images \- AIP Publishing, 访问时间为 五月 21, 2025， [https://pubs.aip.org/asa/jasa/article/157/1/509/3332129/A-diffusion-based-super-resolution-model-for](https://pubs.aip.org/asa/jasa/article/157/1/509/3332129/A-diffusion-based-super-resolution-model-for)  
61. Data augmentation via diffusion model to enhance AI fairness \- Frontiers, 访问时间为 五月 21, 2025， [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1530397/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1530397/full)  
62. Jonathan Ho PhD Google Inc. \- ResearchGate, 访问时间为 五月 21, 2025， [https://www.researchgate.net/profile/Jonathan-Ho-20](https://www.researchgate.net/profile/Jonathan-Ho-20)