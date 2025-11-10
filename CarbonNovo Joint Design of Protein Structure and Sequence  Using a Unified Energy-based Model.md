# CarbonNovo: Joint Design of Protein Structure and Sequence  Using a Unified Energy-based Model

# 摘要

从头蛋白质设计的目标是创造自然界中不存在的全新蛋白质结构和序列。最近，以结构为导向的设计方法通常采用两阶段策略：结构设计模块和序列设计模块被分别训练，并在推理阶段按顺序生成骨架结构和序列。虽然基于扩散的生成模型（如 RFdiffusion）在结构设计方面展现出巨大潜力，但它们在两阶段框架内面临着固有的局限性。首先，序列设计模块可能会过拟合，导致生成的结构与用于训练的晶体结构不一致。其次，序列设计模块缺乏与结构设计模块的交互，无法进一步优化生成的结构。

为了解决这些挑战，我们提出了 **CarbonNovo**——一个统一的能量驱动模型，用于联合生成蛋白质结构和序列。具体而言，我们利用一个基于分数的生成模型和马尔可夫随机场来描述蛋白质结构和序列的能量景观。在 CarbonNovo 中，结构和序列设计模块在每个扩散步骤中进行通信，从而鼓励生成更具一致性的结构-序列对。此外，该统一框架允许将蛋白质语言模型作为进化约束融入到生成的蛋白质中。严格的评估表明，CarbonNovo 在包括可设计性、新颖性、序列合理性以及 Rosetta 能量在内的多种指标上，均优于两阶段方法。

# 背景

蛋白质在生命活动中起关键作用，而**从头蛋白质设计**旨在创造具有特定功能的新蛋白质，广泛应用于药物开发和酶工程。传统方法依赖定向进化或基于几何与能量函数的理性设计。

近年来，**基于扩散的生成模型**（如 RFdiffusion、Chroma、FrameDiff、Flow 模型等）显著推动了蛋白质结构生成的发展，通常采用**“结构生成—序列设计”两阶段框架**。

然而，这种分阶段方法存在两大问题：

1. 结构生成模型输出存在噪声，影响后续序列设计的准确性；
2. 结构与序列模块独立训练，缺乏反馈机制，导致无法协同优化。

因此，**如何在离散序列空间与连续 SE(3) 结构空间中实现联合建模与协同训练**成为当前蛋白质设计领域的核心挑战。

我们提出 **CarbonNovo**，一种统一的能量驱动框架，可**同时生成蛋白质结构与序列**，适用于所有蛋白质家族。CarbonNovo 基于**分数扩散模型与马尔可夫随机场（MRF）\*联合建模蛋白质的结构与序列能量景观，从根本上解决了两阶段设计方法的局限。具体而言，结构与序列模块\*联合训练并在每个扩散步骤中交互**，使二者能够在训练与推理过程中**相互优化**。此外，框架可通过网络循环机制无缝集成如 **ESM-2** 等大型蛋白质语言模型，以引入进化约束，进一步提升生成结果的结构合理性与功能相关性。

我们的主要贡献总结如下：

*   我们开发了 **CarbonNovo**，这是一个能够为通用蛋白质家族同时生成序列和结构的统一框架。
*   我们是第一个将蛋白质语言模型整合到生成过程中，以增强蛋白质结构和序列生成效果的研究。
*   我们探索了多种用于联合模型高效训练和推理的技术，例如多阶段训练策略和用于序列采样的离散版 M-H Langevin 算法。
*   CarbonNovo 在包括可设计性、新颖性、Rosetta 能量和序列合理性在内的各项指标上，均表现出优于两阶段方法的性能。

# 方法

## 3.1 从头蛋白质结构与序列设计的预备知识

CarbonNovo 的目标是联合设计蛋白质主链结构和序列。我们使用 $s \in \mathbb{R}^{N \times 20}$ 来表示氨基酸序列，其中 $N$ 是蛋白质中氨基酸的数量，每个氨基酸有 20 种类型。我们使用 $x \in \mathbb{R}^{N \times 4 \times 3}$ 来表示蛋白质主链原子的 3D 坐标，强调主链由四个原子 $\{C, C_\alpha, N, O\}$ 组成。

我们采用 AlphaFold2（Jumper 等，2021）和 FrameDiff（Yim 等，2023b）中使用的主链框架参数化方法。每个框架 $\mathbf{T} = (\mathbf{R}, \mathbf{t})$ 包含一个旋转 $\mathbf{R} \in \text{SO}(3)$ 和一个平移 $\mathbf{t} \in \mathbb{R}^3$。旋转 $\mathbf{R}$ 由 $C_\alpha$、$N$ 和 $C$ 原子的相对位置决定，而平移 $\mathbf{t}$ 对应于 $C_\alpha$ 原子的坐标。主链上的氧原子则通过一个额外的扭转角 $\phi$ 进行参数化，该角度描述了 $C_\alpha$ 与 $C$ 原子之间键的旋转。

## 3.2 在能量基模型（EBMs）中联合建模结构与序列

由能量基模型（EBM）给出的概率密度可以写作：

$$
p_\theta(\mathbf{a}) = \frac{1}{Z_\theta} \exp[-E_\theta(\mathbf{a})].
$$

这里，$\mathbf{a}$，$-E_\theta(\mathbf{a})$ 和 $Z_\theta$ 分别代表一个单一变量、一个可学习的神经网络和一个归一化因子。

为了生成更具一致性的结构-序列对，我们构建了一个用于蛋白质结构和序列的联合能量框架。蛋白质结构 $\mathbf{T}^{(0)}$ 和序列 $\mathbf{s}$ 的联合分布定义为：

$$
p_\theta(\mathbf{T}^{(0)}, \mathbf{s}) = p_\theta(\mathbf{s} | \mathbf{T}^{(0)}) \; p_\theta(\mathbf{T}^{(0)}) \quad (2)
$$

因此，结构和序列的联合能量表示为：

$$
E(\mathbf{T}^{(0)}, \mathbf{s}) = E_{\text{str}}(\mathbf{T}^{(0)}) + E_{\text{seq}}(\mathbf{s}; \mathbf{T}^{(0)}) \quad (3)
$$

### 结构能量

基于分数的扩散模型已被公认为是一种能量基模型（Du 等，2023）。我们利用 SE(3) 基于分数的扩散模型来表征结构的能量 $E_{\text{str}}(\mathbf{T}^{(0)})$。结构能量可以表达为（Salimans & Ho, 2021; Liu 等，2022a; Du 等，2023）：

$$
E_{\text{str}}(\mathbf{T}^{(0)}) = - \int \mathcal{S}_\theta^{\text{SE}(3)}(\mathbf{T}, t) dt \quad (4)
$$

这里，$\mathcal{S}_\theta^{\text{SE}(3)}(\mathbf{T}, t) = \{\mathcal{S}_\theta^\mathbf{R}(\mathbf{R}, t), \mathcal{S}_\theta^\mathbf{t}(\mathbf{t}, t)\}$ 表示相应分布的分数。

### 序列能量

对于序列能量 $E_{\text{seq}}(\mathbf{s})$，我们采用马尔可夫随机场（MRF）模型，这是一种在蛋白质结构预测（Ekeberg 等，2013；Zhang 等，2023；Ren 等，2024）和蛋白质设计（Ingraham 等，2024）中被广泛使用的能量基模型。

在 MRF 模型下的序列能量定义为：

$$
E_{\text{seq}}(\mathbf{s}; \mathbf{T}^{(0)}) = - \left[ \sum_i \psi_s(s_i | \mathbf{T}^{(0)}) + \sum_{i,j} \psi_p(s_i, s_j | \mathbf{T}^{(0)}) \right]. \quad (5)
$$

这里，$\psi_s$ 和 $\psi_p$ 分别代表来自 MRF 模型的保守性偏置项和成对耦合项。$\mathbf{T}^{(0)}_0$ 是由分数网络预测的最终结构。

## 3.3 模型架构

![image-20251107100755386](C:/Users/wak/AppData/Roaming/Typora/typora-user-images/image-20251107100755386.png)

CarbonNovo 网络由两个主要组件构成：结构设计模块和序列设计模块（图1）。在每个时间步 $t$，网络以噪声主链结构 $\mathbf{T}^{(t)}$ 作为输入，并输出经过优化的主链结构 $\mathbf{T}^{(t-\Delta t)}$ 以及该步骤的最优序列。

### 结构设计模块

与 FrameDiff（Yim 等，2023b）等先前工作不同，后者仅依赖不变点注意力（IPA）网络进行结构模块的设计，CarbonNovo 还额外引入了来自 Evoformer（Jumper 等，2021）的三角注意力网络。

![image-20251107100849553](C:/Users/wak/AppData/Roaming/Typora/typora-user-images/image-20251107100849553.png)

结构设计模块的输入特征包括时间步嵌入、噪声结构 $\mathbf{T}^{(t)}$ 的距离图和框架表示 $(\mathbf{R}^{(t)}, \mathbf{t}^{(t)})$，以及循环特征。

与直接预测分数和优化后的结构 $\mathbf{T}^{(t-\Delta t)}$ 不同，我们预测最终结构 $\hat{\mathbf{T}}^{(0)}$，从中可以推导出分数和 $\mathbf{T}^{(t-\Delta t)}$（附录 B）。这种方法有两个优势：首先，它允许我们在训练过程中通过辅助损失对最终结构施加更多约束（Yim 等，2023b；Watson 等，2023）。其次，它使我们能够在该步骤从最终结构中预测最优序列。

### 序列设计模块

对于序列设计模块，我们也采用了三角注意力网络（附录算法2），该网络改编自 Evoformer（Ren 等，2024；Jumper 等，2021）。

![image-20251107101048138](C:/Users/wak/AppData/Roaming/Typora/typora-user-images/image-20251107101048138.png)

序列设计模块的输入特征包括来自结构设计模块的单体和成对表示、预测的主链结构 $\hat{\mathbf{T}}^{(0)}$ 的直方图，以及循环特征。

MRF 模型中的保守性偏置项和成对耦合项（公式5）随后使用来自序列设计模块更新后的单体和成对表示进行参数化。在训练过程中，我们采用复合似然近似来优化 MRF 模型（Ren 等，2024；Zhang 等，2019；Ingraham 等，2023）。在推理过程中，我们使用离散 Langevin 采样方法从 MRF 模型生成序列（Zhang 等，2022）。

### 网络循环与预训练语言模型

我们从 AlphaFold 用于结构预测的网络循环机制（Jumper 等，2021）和 CarbonDesign 用于蛋白质序列设计的机制（Ren 等，2024）中汲取灵感，将网络循环机制应用于蛋白质结构和序列的协同设计。这种方法具有两大主要优势：首先，它在不增加模型规模的情况下增强了模型容量。其次，它允许从中间预测中提取额外特征，并为后续迭代提供错误反馈。

在 CarbonNovo 中，我们从结构设计和序列设计模块的中间预测中提取额外特征。具体而言，对于结构设计模块，我们提取预测的 $\hat{\mathbf{T}}^{(0)}$ 的距离图和更新后的成对表示，作为后续循环阶段的附加特征。对于序列设计模块，我们提取从 MRF 模型采样的中间序列的语言模型嵌入。这些循环特征被用来更新结构设计和序列设计模块的输入单体和成对表示，如下所示：

$$
\mathbf{r}^s = \mathbf{r}^s + \text{Linear}\left(\text{pLMEmbedding}(\mathbf{s})\right) + \mathbf{r}^s_{\text{prev}},\\
\mathbf{r}^p = \mathbf{r}^p + \text{Linear}\left(\text{DistanceMap}(\mathbf{T}^{(0)}_0)\right) + \mathbf{r}^p_{\text{prev}}.
$$

## 3.4 采样

在结构采样中，我们采用标准的 Langevin 采样算法，这是一种在基于分数的扩散模型中被广泛使用的方法（Song 等，2020b；Yim 等，2023b）。对于序列采样，我们研究了离散的 Metropolis-Hastings Langevin 算法（Zhang 等，2022）。

### 3.4.1 结构采样

遵循 FrameDiff（Yim 等，2023b），我们采用 Langevin 动力学来采样主链结构。

首先，初始结构 $\mathbf{T}^{(T_F)} = (\mathbf{R}^{(T_F)}, \mathbf{t}^{(T_F)})$ 的采样方式如下：

$$
p_{\text{inv}}^{\text{SE}(3)}(\mathbf{T}^{(T_F)}) = P_\#(\mathcal{N}(0, \text{Id}_3)^{\otimes N}) \otimes (\mathcal{I}\mathcal{G}_{\text{SO}(3)}(0, \text{Id}))^{\otimes N}. \quad (7)
$$

然后，在 Langevin 采样过程中，我们利用结构模块 $\mathcal{S}_\theta^{\text{SE}(3)}(\mathbf{T}, t) = \{\mathcal{S}_\theta^\mathbf{R}(\mathbf{R}, t), \mathcal{S}_\theta^\mathbf{t}(\mathbf{t}, t)\}$ 来计算 $\nabla E_{\text{str}}$。结构提议分布可定义为：

$$
q_{\text{str}}(\mathbf{T}^{(t-\Delta t)} | \mathbf{T}^{(t)}) = q_{\text{str}}(\mathbf{R}^{(t-\Delta t)} | \mathbf{R}^{(t)}) q_{\text{str}}(\mathbf{t}^{(t-\Delta t)} | \mathbf{t}^{(t)}),\\
q_{\text{str}}(\mathbf{R}^{(t-\Delta t)} | \mathbf{R}^{(t)}) \sim \mathcal{I}\mathcal{G}_{\text{SO}(3)}(\Delta t \mathcal{S}_\theta^\mathbf{R}(\mathbf{R}^{(t)}), \Delta t \text{Id})^{\otimes N},\\
q_{\text{str}}(\mathbf{t}^{(t-\Delta t)} | \mathbf{t}^{(t)}) \sim \mathcal{P}\mathcal{N}(\mu_\theta, \Delta t \text{Id}_3)^{\otimes N},\\
\mu_\theta = \frac{1}{2} \Delta t \cdot \mathbf{t}^{(t)} + \Delta t \cdot \mathcal{S}_\theta^\mathbf{t}(\mathbf{t}^{(t)}).
$$

这里，$P \in \mathbb{R}^{3N \times 3N}$ 是移除质心的投影矩阵，其中质心为 $\frac{1}{N} \sum_{i=1}^{N} \mathbf{t}_i$，而 $N$ 是设计蛋白质的长度。

### 3.4.2 序列采样

我们采用离散的 Metropolis-Hastings Langevin 采样方法进行序列采样（Zhang 等，2022）。在这里，我们使用上标 $t$ 表示扩散迭代的步骤，下标 $k$ 表示 M-H 采样过程中的步数。

我们仅从单体表示 $\mathbf{r}^s$ 获得初始序列 $\mathbf{s}^{(t)}_{(0)}$。序列提议分布 $q_{\text{seq}}(\mathbf{s}^{(t)}_{(k+1)} | \mathbf{s}^{(t)}_{(k)}, \mathbf{T}^{(0)}_\theta)$ 定义如下：

$$
q_{\text{seq}}(\mathbf{s}^{(t)}_{(k+1)} | \mathbf{s}^{(t)}_{(k)}, \mathbf{T}^{(0)}_\theta) \sim \text{Categorical}(\mathbf{M}^{\text{seq}}),\\
\mathbf{M}^{\text{seq}} = \text{Softmax}\left( \frac{1}{2} \nabla E_{\text{seq}}(\mathbf{s}^{(t)}_{(k)}, \mathbf{T}^{(0)}_\theta) \Delta \mathbf{s} - \frac{\Delta \mathbf{s}^2}{2\gamma} \right),\\
\Delta \mathbf{s} = \mathbf{s}^{(t)}_{(k+1)} - \mathbf{s}^{(t)}_{(k)}.
$$

## 3.5 训练

CarbonNovo 与 RFdiffusion 和 FrameDiff 等先前两阶段方法的一个关键区别在于，它对结构设计模块和序列设计模块进行联合训练。这种联合训练使得序列设计模块的错误反馈能够传递给结构设计模块，从而增强了整体设计过程。

### 3.5.1 训练损失

#### 结构设计损失

在训练结构设计模块时，我们采用了 FrameDiff 的方法（Yim 等，2023b）来获得噪声结构：

$$
d\mathbf{T}^{(t)} = \begin{bmatrix} 0 \\ -\frac{1}{2} P\mathbf{t}^{(t)} \end{bmatrix} dt + \begin{bmatrix} d\mathbf{B}_{\text{SO}(3)^N}^{(t)} \\ P d\mathbf{B}_{\mathbb{R}^{3N}}^{(t)} \end{bmatrix}.
$$

这里，$\mathbf{B}_{\text{SO}(3)^N}^{(t)}$ 和 $\mathbf{B}_{\mathbb{R}^{3N}}^{(t)}$ 分别表示 SO(3) 空间和 $\mathbb{R}^{3N}$ 空间上的布朗运动。

结构设计模块的主要训练目标是去噪分数匹配（DSM）损失（Song 等，2020b；Yim 等，2023b）。DSM 损失 $\mathcal{L}_{\text{dsm}}$（公式19）被分为两个部分：SO(3) 空间中的旋转损失 $\mathcal{L}_{\text{rot}}$ 和 $\mathbb{R}^3$ 空间中的平移损失 $\mathcal{L}_{\text{trans}}$。我们还采用了 FrameDiff 中的辅助损失，包括主链误差损失 $\mathcal{L}_{\text{bb}}$（公式24），以及局部环境内（6Å 范围内）成对原子距离的损失，记为 $\mathcal{L}_{\text{2D}}$（公式23）。

此外，我们使用 FAPE 损失 $\mathcal{L}_{\text{FAPE}}$（公式22），直接监督主链结构的**框架（frames）**，这是一种在蛋白质结构预测任务中已被证明有效的损失函数（Jumper 等，2021）。我们还引入了距离图损失 $\mathcal{L}_{\text{dist}}$（公式21），以直接监督成对表示 $\mathbf{r}^p_{ij}$。更多关于训练损失的细节请参见附录 D.2。

#### 序列设计损失

我们使用单体交叉熵损失 $\mathcal{L}_{\text{single}}$ 和成对交叉熵损失 $\mathcal{L}_{\text{pair}}$ 来分别监督 MRF 模型（公式5）中的保守性偏置项 $\psi_s(s_i | \mathbf{r}^s_i)$ 和成对耦合项 $\psi_p(s_i, s_j | \mathbf{r}^p_{ij})$。

具体而言，对于 $\mathcal{L}_{\text{single}}$，我们首先从单体表示 $\mathbf{r}^s_i$ 计算 logits，然后计算与原生序列作为标签的交叉熵损失。

对于 $\mathcal{L}_{\text{pair}}$，我们使用复合似然来近似 MRF 模型下序列的完整似然。对于序列中的每个氨基酸对 $(s_i, s_j)$，在给定所有其他氨基酸的条件下，其复合似然定义为：

$$
\mathcal{P}(s_i, s_j | s_{i,j}, \mathbf{r}^s_i, \mathbf{r}^p_{ij})
$$

$$
= \log P(S_i = s_i, S_j = s_j | S_{\neg \{i,j\}} = s_{\neg \{i,j\}}; \mathbf{r}^s_i, \mathbf{r}^p_{ij})\\
= \log \left\{ \frac{1}{Z_{ij}} \exp \left[ \psi_s(s_i | \mathbf{r}_i^s) + \psi_s(s_j | \mathbf{r}_j^s) + \psi_p(s_i, s_j | \mathbf{r}_{ij}^p) \right] + \sum_{k \notin \{i,j\}} [ \psi_p(s_i, s_k | \mathbf{r}_{ik}^p) + \psi_p(s_j, s_k | \mathbf{r}_{jk}^p) \right\}.
$$

这里，$Z_{ij}$ 代表归一化因子。我们根据成对表示 $\mathbf{r}^p_{ij}$ 使用复合似然计算氨基酸对的分布，然后计算与原生氨基酸对作为标签的交叉熵损失。

### 3.5.2 训练策略

为了提高训练效率，我们首先分别预训练结构设计模块和序列设计模块。随后，两个模块在 CarbonDesign 框架内以端到端的方式进行联合训练。

#### 预训练阶段

我们使用以下损失函数训练结构设计模块：

$\begin{cases}
\mathcal{L}_{\text{str}} = \mathcal{L}_{\text{dsm}} + \mathcal{L}_{\text{aux}} \mathbb{I}(t < 0.25), \\
\mathcal{L}_{\text{aux1}} = 0.5 \mathcal{L}_{\text{dist}} + 1.0 \mathcal{L}_{\text{bb}} + 1.0 \mathcal{L}_{\text{2D}} + 2.0 \mathcal{L}_{\text{FAPE}}, \\
\mathcal{L}_{\text{dsm}} = 1.0 \mathcal{L}_{\text{trans}} + 0.5 \mathcal{L}_{\text{rot}}.
\end{cases}$

我们仅对 $t < 0.25$ 的样本应用辅助损失函数（Yim 等，2023b）。

我们使用以下损失函数训练序列设计模块：

$\mathcal{L}_{\text{seq}} = 1.0 \mathcal{L}_{\text{single}} + 1.0 \mathcal{L}_{\text{pair}} + 0.01 \mathcal{L}_1 + 0.02 \mathcal{L}_2.$

这里，$\mathcal{L}_1$ 和 $\mathcal{L}_2$ 分别表示用于参数化 MRF 模型（公式5）的单体和成对表示的 L1 和 L2 正则化项。

在训练序列设计模块时，我们向晶体结构添加噪声以缓解过拟合。具体而言，我们在晶体主链结构上采用扩散过程的前向过程，其中添加噪声的时间步长 $t$ 服从均匀分布 $t \sim \text{Uniform}([0, 0.1])$（公式10）。

此阶段包括结构设计模块的 10k 训练步骤和序列设计模块的 9k 训练步骤。

#### 联合训练阶段

对于此阶段，我们从预训练阶段初始化模型权重。我们注意到，在序列设计模块的预训练阶段，输入的单体表示被设为 0；而在联合训练阶段，它们被设为结构设计模块输出的单体表示。

联合训练阶段包含两个子阶段。在第一阶段，我们添加一个侧链预测目标作为辅助损失，以同时优化蛋白质序列和主链结构（Jumper 等，2021；Ren 等，2024）。该损失仅应用于 $t < 0.05$ 的训练样本。在第二阶段，我们引入一个碰撞损失（clash loss），以消除局部结构构象中的空间位阻。此外，我们将第二阶段的裁剪尺寸从 256 扩大到 320。

$\begin{cases}
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{dsm}} + \mathcal{L}_{\text{aux}} \mathbb{I}(t < 0.25) + 0.01 \mathcal{L}_{\text{seq}} \mathbb{I}(t < 0.1) \\
\quad + 0.1 \mathcal{L}_{\text{side}} \mathbb{I}(t < 0.05), \\
\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{dsm}} + \mathcal{L}_{\text{aux}} \mathbb{I}(t < 0.25) + 0.1 \mathcal{L}_{\text{seq}} \mathbb{I}(t < 0.1) \\
\quad + 0.1 \mathcal{L}_{\text{side}} \mathbb{I}(t < 0.05) + 0.1 \mathcal{L}_{\text{viol}} \mathbb{I}(t < 0.05).
\end{cases}$

第一和第二子阶段分别涉及 100k 和 10k 训练步骤。