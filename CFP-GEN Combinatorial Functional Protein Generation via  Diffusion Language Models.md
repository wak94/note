# CFP-GEN: Combinatorial Functional Protein Generation via  Diffusion Language Models

# 摘要

现有的蛋白质语言模型（PLMs）通常仅基于单一模态的单一条件约束来生成蛋白质序列，难以同时满足跨不同模态的多重约束。在本研究中，我们提出了 **CFP-GEN**（Combinatorial Functional Protein GENeration），一种新颖的扩散语言模型，用于组合功能性蛋白质的从头设计。CFP-GEN 通过融合功能、序列和结构等多模态约束，实现对蛋白质生成过程的综合引导。

具体而言，我们引入了两个关键模块：  
1. **注释引导的特征调制模块**（Annotation-Guided Feature Modulation, AGFM）：该模块可根据可组合的功能注释（如 Gene Ontology 术语、InterPro 结构域和 EC 编号）动态调整蛋白质特征分布，从而实现对复杂功能语义的灵活编码；  
2. **残基控制的功能编码模块**（Residue-Controlled Functional Encoding, RCFE）：该模块建模残基级别的相互作用，以确保对序列功能的精细控制。

此外，CFP-GEN 可无缝集成现成的三维结构编码器，以施加几何约束，进一步提升生成蛋白质的结构合理性。

实验表明，CFP-GEN 能够高效生成大量新颖蛋白质，其功能特性可与天然蛋白质相媲美，并在多功能蛋白质设计任务中展现出高成功率。

# 背景

**从头蛋白质设计**已成为药物开发、酶工程和新型治疗蛋白创制的核心策略。尽管大型蛋白质语言模型（PLMs）展现出生成全新蛋白质的巨大潜力，现有方法多局限于**无条件或单条件生成**，难以高效满足现实场景中**跨模态的多重功能约束**（如功能注释、序列基序与三维结构）。

![image-20251020201216094](./CFP-GEN%20Combinatorial%20Functional%20Protein%20Generation%20via%20%20Diffusion%20Language%20Models.assets/image-20251020201216094.png)

为此，我们提出 **CFP-GEN**——一种面向**组合功能性蛋白质生成**的大规模扩散语言模型。CFP-GEN 在统一框架内协同整合多模态条件：  
- 通过 **注释引导特征调制（AGFM）** 模块，以可组合方式动态调制去噪过程，实现功能标签（如 GO、EC、IPR）与序列的严格对齐；  
- 引入 **残基控制功能编码器（RCFE）**，显式建模关键功能基序及其上位效应，生成功能完备的新序列；  
- 支持以**用户提供的三维骨架坐标**为条件，实现结构引导的逆折叠，兼顾功能与结构保真度。

实验表明，CFP-GEN 在多项任务中显著领先：功能预测 F₁ 分数较 ESM3 提升 **30%**，逆折叠氨基酸恢复率（AAR）较 DPLM 提高 **9%**，并在设计多功能酶等复杂任务中展现出卓越成功率。CFP-GEN 为多目标蛋白质工程提供了一种高效、灵活且实用的计算范式。

# 3. 方法

## 3.1 CFP-GEN 的预备知识

扩散模型因其在从头蛋白质设计中的能力而被广泛认可。CFP-GEN 建立在领先的扩散蛋白质语言模型 DPLM 之上，旨在利用其从进化规模数据集中获得的预训练参数。具体而言，我们采用离散扩散机制来建模氨基酸类别层面的蛋白质序列分布。

设 $\mathbf{x} \sim q(\mathbf{x})$ 表示一个长度为 $L$ 的蛋白质序列，表示为 $\mathbf{x}=(x_1,x_2,...,x_L)$，其中每个 $x_i \in \{ 0, 1 \}^{|V|}$ 是一个独热向量，用于指示来自包含 20 种标准类型的集合 $V$ 中的一个氨基酸类别。分类分布 $\text{Cat}(\mathbf{x}; \mathbf{p})$ 对序列 $\mathbf{x}$ 进行建模，其中 $\mathbf{p} = (\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_L)$ 是概率向量的集合。每个 $\mathbf{p}_i = (p_{i,1}, p_{i,2}, ..., p_{i,|V|})$ 指定了序列中第 $i$ 个残基的分类分布，即 $p_{i,v}$ 表示选择氨基酸类别 $v$ 的概率，并始终确保对所有 $i \in \{1, ..., L\}$，满足 $\sum_{v \in V} p_{i,v} = 1$。

### 离散扩散的前向过程

我们应用离散扩散，通过在 $t \in \{1, \dots, T\}$ 个时间步内，将每个氨基酸标记逐步过渡到一个平稳噪声分布，从而逐渐污染原始序列 $\mathbf{x}^{(0)}$。该平稳噪声分布由一个固定的概率向量 $\mathbf{q}_{\text{noise}}$ 参数化，可表示为：$q_{\text{noise}}(\mathbf{x}^{(t)}) = \text{Cat}(\mathbf{x}^{(t)}; \mathbf{p} = \mathbf{q}_{\text{noise}})$。遵循 DPLM 的做法，$q_{\text{noise}}(\mathbf{x}^{(t)})$ 满足：当 $\mathbf{x}^{(t)} = [X]$ 时，$q_{\text{noise}}(\mathbf{x}^{(t)}) = 1$，否则为 0，其中 $[X]$ 指代吸收态（例如，$\langle mask \rangle$）。一旦一个标记过渡到此吸收态，它将在所有后续的扩散步骤中保持不变。这确保了前向过程最终会使所有氨基酸标记都变为 $\langle mask \rangle$，统一了掩码语言模型和自回归语言模型背后的原理。数学上，前向转移过程由下式给出：

$$
q(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}) = \text{Cat}(\mathbf{x}^{(t)}; \mathbf{p} = \mathbf{x}^{(t-1)} \mathbf{Q}_t)
$$
其中 $\mathbf{Q}_t$ 是第 $t$ 步的转移矩阵。$\mathbf{Q}_t$ 的每一行都是一个概率向量，定义为：$\mathbf{Q}_t = \beta_t I + (1 - \beta_t) \mathbf{q}_{\text{noise}}$，其中 $I$ 是单位矩阵，$\beta_t \in [0, 1]$ 是噪声调度。由于马尔可夫性质，从 $\mathbf{x}^{(0)}$ 到 $\mathbf{x}^{(t)}$ 的整体转移可以表示如下：

$$
q(\mathbf{x}^{(t)} \mid \mathbf{x}^{(0)}) = \text{Cat}(\mathbf{x}^{(t)}; \mathbf{p} = \alpha_t \mathbf{x}^{(0)} + (1 - \alpha_t) \mathbf{q}_{\text{noise}}), \quad (2)
$$

其中 $\alpha_t = \prod_{i=1}^{t} \beta_i$ 表示噪声调度在 $t$ 步内的累积效应。当 $t \to T$ 时，$\alpha_t \to 0$，从而确保序列 $\mathbf{x}^{(t)}$ 在时间步 $T$ 收敛到平稳噪声 $\mathbf{q}_{\text{noise}}$。

### 带可组合条件的逆向去噪

![image-20251020201314894](./CFP-GEN%20Combinatorial%20Functional%20Protein%20Generation%20via%20%20Diffusion%20Language%20Models.assets/image-20251020201314894.png)

为了实现具有期望功能性的蛋白质序列生成，我们在逆向扩散过程中将多模态条件融入离散扩散框架，如图 2 所示。这些条件包括：0维注释标签 $c_{\text{anno}}$、1维序列基序 $c_{\text{seq}}$ 和3维结构 $c_{\text{str}}$，它们分别由函数 $f_{\text{AGFM}}$、$f_{\text{RCFE}}$ 和 $f_{\text{GVPT}}$ 进行编码。这些网络的细节将在后续章节中介绍，其中 GVPT 指的是用于编码三维骨架原子坐标的 GVP-Transformer。

逆向过程通过迭代地对 $\mathbf{x}^{(t)}$ 进行去噪来重建序列至 $\mathbf{x}^{(0)}$，在每一步中使用预测的 $\hat{\mathbf{x}}^{(0)}$，该预测值源自 KL 散度，即 $D_{\text{KL}} \left[ q(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}, \mathbf{x}^{(0)}) \parallel p_\theta(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}) \right]$。因此，逆向步骤表示如下：

$$
p_\theta(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}) = \sum_{\hat{\mathbf{x}}^{(0)}} q(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}, \hat{\mathbf{x}}^{(0)}) p_\theta(\hat{\mathbf{x}}^{(0)} \mid \mathbf{x}^{(t)}, c)
$$

其中 $c \in \{c_{\text{anno}}, c_{\text{seq}}, c_{\text{str}}\}$ 代表根据其可用性而定的任意条件组合。为预测 $\hat{\mathbf{x}}^{(0)}$，模型在整个网络中整合了条件 $c$：

$$
p_\theta(\hat{\mathbf{x}}^{(0)} \mid \mathbf{x}^{(t)}, c) = \text{Softmax}(W h(\mathbf{x}^{(t)}, c))\\
h(\mathbf{x}^{(t)}, c) = \mathcal{A} \Big( f_{\text{RCFE}} \big( f_{\text{AGFM}}(\mathbf{x}^{(t)}, c_{\text{anno}}), c_{\text{seq}} \big), f_{\text{GVPT}}(c_{\text{str}}) \Big)
$$

其中 $W$ 是最终输出层，$\mathcal{A}$ 是一个交叉注意力层，而 $h(\mathbf{x}^{(t)}, c)$ 代表 CFP-GEN 的核心，它按顺序整合了诸如 $f_{\text{AGFM}}$、$f_{\text{RCFE}}$ 和 $f_{\text{GVPT}}$ 等模块。

### CFP-GEN 的优化

训练目标是使用加权交叉熵损失来优化预测的 $\hat{\mathbf{x}}^{(0)}$ 与原始序列 $\mathbf{x}^{(0)}$ 之间的匹配：

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{x}^{(0)})} \left[ \lambda^{(t)} \sum_{1 \le i \le L} b_i(t) \cdot \log p_\theta(\mathbf{x}_i^{(0)} | \mathbf{x}^{(t)}, \gamma(c)) \right]
$$

其中，$\lambda^{(t)}$ 调整每个扩散时间步 $t$ 的影响，$b_i(t)$ 决定每个位置 $i$ 的贡献，而 $\gamma(c)$ 控制条件 $c$ 的强度。在后续章节中，我们将展示条件 $c$ 的引入如何促进高度功能性蛋白质的预测，以及其可组合的特性如何赋予 CFP-GEN 实现多目标蛋白质设计的能力。

## 3.2 基于 AGFM 的注释引导条件化

功能注释因其由生物学家精心策划，能捕捉蛋白质的核心特性，故而信息丰富且具有代表性。在本工作中，我们考虑三种常用的注释——基因本体（GO）术语、InterPro 编号和酶分类（EC）编号——来引导生成过程。通常，每种类型的注释从不同角度刻画蛋白质的特征：GO 描述分子功能，IPR 描述功能域，而 EC 描述相关的催化过程。

与仅依赖受限词汇表映射来表示 IPR 注释的 ESM3 不同，我们采用了一种更巧妙的方法来组合多种功能。虽然我们以这三种注释为例，但我们的范式可扩展至其他形式的注释，例如 Pfam。

现有的蛋白质语言模型（PLMs）通常只编码一种类型的注释。例如，ZymCTRL 仅基于 EC 编号设计酶，而 ProteoGAN 仅操作 GO 术语。尽管 ProGen 支持多种注释标签，但它需要对大量同源序列进行广泛的微调以确保性能，这限制了其在训练样本稀缺时的应用。相比之下，CFP-GEN 更加灵活和通用，能够同时处理多种注释，无需进一步微调。通过利用这些高质量注释的互补特性，我们还能实现对蛋白质更全面的描述。

如图 2 所示，注释引导特征调制（AGFM）模块被集成到每个修改后的 ESM 块中，其中每个块均已预先由 DPLM 训练好。CFP-GEN 通过训练注释嵌入层和一个单一的 MLP 层来实现。对于每种类型的注释（例如，GO、IPR、EC），我们维护一个专用的嵌入层。每种注释都被映射到一个相同维度的向量表示，并通过其各自的嵌入层。这种灵活的设计确保了来自不同来源的注释可以直接相加。

该过程的形式化表示为：$\mathbf{x}_{\text{anno}} = f_{\text{AGFM}}(\mathbf{x}, c_{\text{GO}}, c_{\text{IPR}}, c_{\text{EC}})$，其中 $\mathbf{x}, \mathbf{x}_{\text{anno}} \in \mathbb{R}^{L \times D}$ 分别表示输入特征和在时间步 $t$ 基于去噪序列 $\mathbf{x}^{(t)}$ 由 AGFM 在单个块内调制后的特征输出。具体而言，嵌入的总和会经过一个 MLP 层进行重组，即 $\gamma, \beta, \alpha = \mathcal{F}(c_{\text{GO}} + c_{\text{IPR}} + c_{\text{EC}})$，其中 $\gamma, \beta$ 和 $\alpha$ 分别是缩放、偏移和门控因子。

MLP 层 $\mathcal{F}$ 遵循一种专门的初始化策略：预测 $\gamma$ 和 $\beta$ 的权重被初始化为零，以确保条件信息是逐步注入到 $\mathbf{x}^{(t)}$ 中的；而预测 $\alpha$ 的权重则从一开始就初始化，以实现有效的门控。缩放参数 $\gamma$ 和偏移参数 $\beta$ 会在自注意力（SA）层和前馈网络（FFN）层之前调节噪声特征 $\mathbf{x}$ 的分布，类似于特征去归一化过程：$\mathbf{x}_{\text{out}} = \gamma \odot \mathbf{x} + \beta$。此外，门控因子 $\alpha$ 作用于 SA 和 FFN 层输出的特征分布 $\mathbf{x}'_{\text{out}}$，通过以下公式进行计算：$\hat{\mathbf{x}}_{\text{out}} = \alpha \odot \mathbf{x}'_{\text{out}} + \mathbf{x}_{\text{out}}$。

通过不同注释的协作，AGFM 在扩散过程中有效地调整中间表示，从而生成比以往依赖单一条件控制方法质量更高的蛋白质。

## 3.3 基于 RCFE 的序列引导条件化

尽管 AGFM 为蛋白质生成提供了有效的控制，但某些应用需要更精细的指导。例如，生物学家通常会关注功能性的序列片段（即序列基序），以确保生成的蛋白质保留与这些基序相关联的功能性。因此，我们提出了残基控制功能编码器（RCFE）来处理这种序列级条件。

RCFE 的核心在于推断完整的序列，同时保留甚至优化指定的序列基序，这类似于 DPLM 和 EvoDiff 中引入的序列填充任务。然而，与这些依赖于生成过程中固定氨基酸的方法不同，RCFE 在推理时动态更新指定的基序。这种动态调整使模型能够模拟进化过程，从而有潜力发现具有增强功能特性的改进序列基序。

具体而言，公式(4)中的 $f_{\text{RCFE}}$ 由两个分支组成：修改后的 ESM 块以及零初始化的线性层，共同处理序列级和注释级信息。如图 2 所示，底部的灰色块代表主干分支，负责处理被噪声污染的序列和注释标签；而上方的橙色块代表第二分支，由主干分支的可训练副本构成，用于处理序列级条件化。受 ControlNet 的启发，RCFE 将这种双分支设计适配到基于 Transformer 的 ESM 块中，使其非常适合蛋白质序列生成。

设 $\mathcal{E}_{\text{esm}}(\cdot, \Theta_{\text{esm}})$ 表示经过 AGFM 增强、已整合功能注释的主干修改版 ESM 块。在训练 RCFE 时，主干块中的参数 $\Theta_{\text{esm}}$ 保持冻结，而另一个分支中的参数 $\Theta_{\text{seq}}$ 则使用可训练副本进行更新，记为 $\mathcal{E}_{\text{seq}}(\cdot, \Theta_{\text{seq}})$。该设计保留了已完全训练好的主干分支强大的表示能力，同时使另一分支能够动态地编码序列基序。

形式上，我们首先用 `<mask>` 填充序列基序，并将其投影到与去噪特征 $\mathbf{x} \in \mathbb{R}^{L \times D}$ 相同的潜在空间内，这被称为 $\mathbf{c}_{\text{seq}} \in \mathbb{R}^{L \times D}$。这样，RCFE 处理序列级条件的工作流程可以表示如下：

$$
\mathbf{x}_{\text{seq}} = \mathcal{E}_{\text{esm}}(\mathbf{x}; \Theta_{\text{esm}}) + F_{\text{out}}(\mathcal{E}_{\text{seq}}(\mathbf{x} + F_{\text{in}}(\mathbf{c}_{\text{seq}}; \Theta_{\text{in}}); \Theta_{\text{seq}}); \Theta_{\text{out}})
$$

其中，$F_{\text{in}}(\cdot; \Theta_{\text{in}})$ 表示应用于 $\mathbf{c}_{\text{seq}}$ 的一个零初始化线性层。请注意，该层仅在条件分支的第一个块中使用。类似地，$F_{\text{out}}(\cdot; \Theta_{\text{out}})$ 是另一个零初始化线性层。零初始化使得有意义的信息能够逐步融入，同时确保主干块的稳定性。最终，$\mathbf{x}_{\text{seq}}$ 代表了从主干块获得的、在注释和序列基序层面都得到丰富条件信息的更新特征。

值得一提的是，在图像生成模型中，U-Net 的编码器部分通常用作可训练分支。类似地，我们仅利用 ESM2 模型的前半部分块来编码序列基序，以确保高效且富有表现力的表示。

因此，序列级条件作为一个强有力的补充，与注释级条件相结合，显著增强了对生成序列的可控性，相较于仅使用单一条件模态的方法效果更佳。

## 3.4 CFP-GEN 中的多模态条件化

蛋白质科学的基本原则是功能、序列和结构本质上是相互关联的。这种错综复杂的关系揭示了联合建模这些模态对于实现精确蛋白质设计的重要性。

为此，我们考虑一个实际场景：生物学家已经获得了感兴趣的骨架结构（例如，来自 RFDiffusion），并希望确定能折叠成该结构的功能性序列——这一问题被称为“逆折叠任务”。在传统范式中，目标是生成一个能正确折叠成给定骨架结构的序列。然而，当需要额外的功能约束 $c_{\text{anno}}$ 时，此任务可以扩展为“功能性蛋白质逆折叠”或“逆功能任务”，即生成一个序列 $\mathbf{x}^*$，它在最大化序列恢复的同时，也能优化与 $c_{\text{anno}}$ 匹配的功能性：

$$
\mathbf{x}^* = \arg\max_{\mathbf{x}} p(\mathbf{x} \mid c_{\text{anno}}, c_{\text{seq}}, c_{\text{str}}) + S_{\text{func}}(\mathbf{x}, c_{\text{anno}})
$$

其中，$S_{\text{func}}$ 是一个评分函数，用于评估生成的序列 $\mathbf{x}$ 与期望功能属性的匹配程度。关于 $S_{\text{func}}$ 的更多讨论见 §4.2。

为实现公式(7)的目标，我们采用结构编码器 $f_{\text{GVPT}}$（即 GVP-Transformer）来嵌入蛋白质的骨架原子坐标：$\mathbf{c}_{\text{str}} = f_{\text{GVPT}}(c_{\text{str}})$。随后，$\mathbf{c}_{\text{str}}$ 通过交叉注意力层注入到主干分支的最后一个 ESM 块中，遵循 DPLM 的策略。在我们的实现中，我们发现 DPLM 的预训练交叉注意力层可以直接使用，无需进一步训练，这表明可以集成更多的功能适配器。已经整合了 $c_{\text{anno}}$ 和 $c_{\text{seq}}$ 的去噪特征，会进一步与 $\mathbf{c}_{\text{str}}$ 交互，从而实现更精确且功能上连贯的蛋白质设计。

我们的功能性蛋白质逆折叠将搜索空间从庞大的序列空间缩小到既在结构上可行又在功能上相关的序列。与仅依赖结构约束的传统逆折叠相比，我们的方法将功能信息作为附加约束，从而实现了更高的序列恢复率。我们在 §4.3 中展示了带有结构适配器的 CFP-GEN 的零样本和监督微调（SFT）版本的显著改进。

# 4. 实验

## 4.1 实验设置

### 数据集

为了收集高质量的数据用于训练 CFP-GEN，我们采用了来自 SwissProt (UniProtKB) (Consortium, 2019)、InterPro (Hunter 等, 2009) 和 CARE (Yang 等, 2024) 数据库的专家策划的功能注释。为此构建了两个数据集，分别用于通用蛋白质设计和酶设计。对于**通用蛋白质数据集**，我们包含了 103,939 条蛋白质序列，覆盖了 375 个 GO 术语和 1,154 个 IPR 结构域。对于**酶设计数据集**，我们将 SwissProt 与 CARE 数据集取交集，得到了 139,551 条带有 661 个 EC 编号（4级 EC 注释）的酶序列。此外，我们利用 PDB (Berman 等, 2000) 和 AFDB (Jumper 等, 2021) 数据库提供骨架原子坐标，确保结构约束被纳入数据集。

### 实现细节

我们的模型基于预训练的 DPLM-650M，并分两个渐进阶段进行训练。

首先，我们训练 AGFM 模块，以实现功能注释的有效整合。第一阶段的预训练参数随后用于下一阶段，在该阶段我们训练 RCFE 中的副本 ESM 块，使模型能够根据序列基序进行条件化。优化调度与无条件 DPLM 相同。对于结构条件化，我们直接使用 DPLM 的 GVP-Transformer 和结构适配器，它们已在 CATH 数据库上预训练，无需为 CFP-GEN 进行额外的微调。在推理时，用户可以指定任何功能约束及其组合作为条件信号。

#### 超参数细节

正如论文中所介绍的，除非另有说明，扩散模型的大部分学习策略和超参数均与 DPLM 保持一致。批大小设置为 100 万个 token，每个阶段的训练在 8 块 NVIDIA A100 GPU 上进行，耗时约 72 小时。采用 AdamW 优化器，最大学习率为 0.00004。在推理阶段，我们允许模型执行 100 个采样步骤，遵循 DPLM 的条件生成方式，序列长度在 200 到 400 之间变化。CFP-GEN 的总模型规模为 14.8 亿（1.48B）参数，不包括 GVP-transformer 结构编码器。

![image-20251020214836239](./CFP-GEN%20Combinatorial%20Functional%20Protein%20Generation%20via%20%20Diffusion%20Language%20Models.assets/image-20251020214836239.png)

在训练期间，我们通过在每个条件输入上应用 dropout 来调整公式 (5) 中的 $\gamma(c)$。随机丢弃每个条件及其对模型性能相应影响的概率如图 8(a) 所示。我们使用 MRR（平均倒数排名）来呈现性能，因为我们发现该指标更具鲁棒性和表现力。结果表明，将每个条件的 dropout 概率设置为 0.5 可以获得最佳性能。此外，我们研究了 RCFE 模块中 ESM 块副本的最优数量。如图 8(b) 所示，将块的数量设置为 16 可以在性能和模型复杂度之间实现最佳权衡。

## 4.2 评估蛋白质功能性能

本节评估生成蛋白质序列的功能保真度。我们基于通用蛋白和酶数据集构建两个验证集，分别用 GO/IPR 和 EC 注释作为条件提示，每条天然序列对应生成一条可变长序列。

评估包括两方面：  
1. **序列相似性**：采用 MMD、MMD-G 和 MRR；  
2. **功能一致性**：使用 DeepGO-SE（GO）、InterProScan（同源注释）和 CLEAN（催化功能）预测标签，并以微/宏 F₁、宏 AUPR 和宏 AUC 衡量预测标签与提示标签的对齐程度。

我们对比了主流 PLMs 与 **CFP-GEN** 的性能。值得注意的是，CFP-GEN 支持跨模态组合条件，而大多数基线仅支持单一条件。正向对照为 UniProtKB 真实序列，负向对照为 DPLM 无条件生成序列，用于界定指标上下限。完整实现细节见附录 §E。