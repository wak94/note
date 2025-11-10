# SurfPro: Functional Protein Design Based on Continuous Surface

# 摘要

我们如何设计具有所需功能的蛋白质？我们的动机源于一个化学直觉：蛋白质的功能既取决于其几何结构，也取决于其生化特性。在本文中，我们提出了 **SurfPro**，这是一种给定期望表面及其相关生化特性后生成功能性蛋白质的新方法。

SurfPro 包含一个分层编码器，它逐步建模蛋白质表面的几何形状和生化特征；以及一个自回归解码器，用于生成氨基酸序列。

我们在标准逆折叠基准测试 CATH 4.2 和两个功能性蛋白质设计任务上评估了 SurfPro：蛋白质结合剂设计和酶设计。我们的 SurfPro 持续优于以往最先进的逆折叠方法，在 CATH 4.2 上实现了 57.78% 的恢复率，并在蛋白质-蛋白质结合和酶-底物相互作用得分方面取得了更高的成功率。

# 背景

![image-20251110153728091](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110153728091.png)

蛋白质是生命系统中执行多种关键细胞功能的核心分子。随着 **生成式人工智能（Generative AI）** 的快速发展，蛋白质设计领域迎来了革命性变化。
 传统的蛋白质设计流程通常分为两步：

1. **确定目标主链结构** —— 即定义蛋白质的几何形状（不含氨基酸类型）；
2. **逆折叠（Inverse Folding）** —— 为给定主链寻找能正确折叠到该结构的氨基酸序列

第一步指定了所需蛋白质的几何形状（不包含氨基酸类型），第二步（也称为逆折叠）如图 1(a) 所示，确定与给定形状相对应的氨基酸组成。这种流程可以有效地生成结构合理的蛋白质序列，但仍存在明显的局限性。

## 研究现状

当前主流的逆折叠方法（如 **ProteinMPNN, PiFold, LM-DESIGN**）能够在几何层面上生成与目标结构匹配的序列，但其控制主要基于**主链几何约束**，缺乏对蛋白质 **功能性要求（如结合能力、催化活性）** 的直接建模。
 然而，蛋白质的功能不仅取决于其三维形状，还与表面上的 **电荷分布、极性、疏水性** 等生化属性密切相关。
 例如，即使两个蛋白质形状互补，它们也可能由于界面电性或极性不匹配而无法有效结合。

## 研究动机

传统逆折叠仅考虑几何约束，**无法实现功能导向的蛋白质设计**。为了让设计的蛋白质具备目标功能（如底物结合、靶向抑制），需要在几何之外引入 **生化属性约束**。

因此，本文提出一种新的蛋白质设计范式：不仅“让序列折叠成目标形状”，而且“让序列具备期望的表面生化特征”。

## 研究贡献

*   我们提出了 SurfPro，用于基于连续表面（并增强了生化属性）设计功能性蛋白质。
*   我们在标准逆折叠基准测试 CATH 4.2 上评估了 SurfPro。SurfPro 实现了 57.78% 的序列恢复率和 3.13 的复杂度，显著优于之前的逆折叠方法，包括 ProteinMPNN (Dauparas 等, 2022)、PiFold (Gao 等, 2022) 和 LM-DESIGN (Zheng 等, 2023)。
*   我们设置了一个结合剂设计任务，并使用 AlphaFold2 (Jumper 等, 2021) 的 pAE 相互作用评分 (Bennett 等, 2023; Watson 等, 2023) 来评估所设计蛋白质的结合能力。SurfPro 展现出更强的能力，能够设计出与目标蛋白结合力更强的结合剂，其在六个目标上的平均成功率为 26.22%，比最佳先前方法高出 6.9%。
*   我们设置了一个酶设计任务，并使用 ESP 评分 (Kroll 等, 2023a,b) 来衡量所设计酶与其底物之间的结合。SurfPro 能够设计出比天然酶具有更高酶-底物相互作用得分的酶，在五个酶数据集上实现了 43.46% 的平均成功率，比最佳先前方法高出 2.98%。

# 方法

![image-20251110154220374](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110154220374.png)

一个分子表面定义了蛋白质在三维欧几里得空间中的形状及其生化属性，如疏水性和电荷。表面形状和相关的生化属性共同决定了蛋白质的潜在功能。给定一个带有几何和生化约束的期望表面，我们如何生成能够贴合该表面的蛋白质序列？在本节中，我们介绍了 SurfPro，这是一种基于蛋白质表面的功能性蛋白质设计新方法。我们的方法作用于蛋白质表面的连续点云表示。SurfPro 包含一个分层编码器，它从局部视角到全局景观逐步建模 3D 几何形状和生化特征；以及一个自回归解码器，它根据相应表面的几何和生化约束生成蛋白质序列。图 2(a) 给出了  SurfPro 的概览。

## 表面生成

![image-20251110154429113](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110154429113.png)

我们的方法作用于蛋白质表面的连续点云。一个高质量的表面应满足以下两个特性：(1) **平滑性**：由点云定义的表面应表现出足够的平滑度；(2) **紧凑性**：点云应去除冗余信息，通过下采样来提高效率。

### 原始表面构建

我们使用 MSMS 来计算蛋白质的原始分子表面，该表面以包含 $N$ 个顶点 $\{x_1, x_2, ..., x_N\} \in \mathbb{R}^{N \times 3}$ 的点云形式提供。假设该蛋白质是一个长度为 $L$ 的氨基酸序列 $y = \{y_1, y_2, ..., y_L\} \in \mathcal{A}^L$，其中 $\mathcal{A}$ 是 20 种常见氨基酸的集合，且 $N \gg L$。我们将每个顶点的生化特征与其最近原子所属的残基相关联。具体而言，我们为每个顶点 $x_i$ 分配两个生化特征：其疏水性 $t_i$ 和电荷 $c_i$。然后，我们根据其最近原子所属残基的索引对所有顶点进行排序。附录图 6(a) 展示了一个原始表面的示例。

### 表面平滑

正如先前方法所述，原始点云通常带有噪声，这可能会限制分子表面的表达能力。因此，点云去噪和平滑是必要的。我们对原始点云数据应用高斯核平滑：

$$
\boldsymbol{x}'_i = \sum_{\boldsymbol{x}_j \in \mathcal{N}(\boldsymbol{x}_i)} \frac{\mathcal{K}(\boldsymbol{x}_i, \boldsymbol{x}_j) \boldsymbol{x}_j}{\sum_{\boldsymbol{x}_t \in \mathcal{N}(\boldsymbol{x}_i)} \mathcal{K}(\boldsymbol{x}_i, \boldsymbol{x}_t)}, \quad \mathcal{K}(x, y) = e^{-\frac{(x-y)^2}{\eta}} \quad (1)
$$

其中 $i \in \{1, 2, ..., N\}$。$x_i$ 和 $x_j$ 分别表示原始点云中第 $i$ 个和第 $j$ 个顶点的坐标。$\mathcal{N}(\boldsymbol{x}_i)$ 是 $\boldsymbol{x}_i$ 的 $K$-近邻。$\mathcal{K}(\cdot, \cdot)$ 是高斯核，其中 $\eta$ 表示点云空间中的距离尺度。在本文中，我们设定 $\eta = \max(\text{dist}(\boldsymbol{x}_i, \mathcal{N}(\boldsymbol{x}_i)))$，其中 $i \in \{1, ..., N\}$。最近邻的数量 $K$ 设为 8。对核平滑的深入分析支持表面是无限平滑的，即 $\theta \in C^\infty$。附录图6(b) 展示了一个平滑表面的示例。

### 表面压缩

为了减少表面点数并提高采样效率，我们使用一种基于八叉树的压缩方法对蛋白质表面进行下采样。我们使用八叉树将表面转换为小立方体，并估计每个立方体的局部密度。

每个八叉树节点被递归地划分为八个相等的八分体。每次划分后，检查每个节点中点的数量，以确定是否继续划分当前节点。那些点数少于特定阈值 $N_{\min}$ 的立方体被视为叶节点，不再进一步划分。在所有节点处理完毕后，点云被转换为一组体积不等的立方体，这些立方体基于点的分布而形成。低密度区域会生成较大的立方体。每个立方体所需的点数为 $N_s = V_s * r$，其中 $V_s$ 是第 $s$ 个立方体中的点数，$r$ 是期望的下采样比率。图6(c) 展示了一个压缩表面的示例。

## 3.2 分层表面编码器

我们设计了一个分层编码器来建模蛋白质表面的几何形状和生化属性。

### 局部视角建模

相邻残基之间表现出强烈的相互作用。为了对表面上最近顶点之间的这种相互作用进行建模，我们设计了一种由 Satorras 等人（2021）提出的等变图卷积层（EGCL）的变体，以捕捉局部几何和生化特征（图2(b) 左侧模块）。具体而言，在表面压缩后，该表面有 $N'$ 个顶点（$N' \leq N$），每个顶点具有一个三维坐标 $x'_i \in \mathbb{R}^3$ 和两个生化特征 $h_i = [t_i, c_i]^T$，其中 $t_i$ 表示其疏水性，$c_i$ 表示其电荷，且 $i \in \{1, ..., N'\}$。我们按如下方式计算局部消息：

$$
\boldsymbol{m}'_{ij} = \text{SiLU}(\phi_e([(\boldsymbol{h}^l_i; \boldsymbol{h}^l_j) || |\boldsymbol{x}'_i - \boldsymbol{x}'_j|_2]))\\
w^l_{ij} = \frac{\exp(W^l_s \boldsymbol{m}'_{ij} + b^l_s)}{\sum_{k \in \mathcal{N}(x_i)} \exp(W^l_s \boldsymbol{m}'_{ik} + b^l_s)}\\
\boldsymbol{m}^{l+1}_{ij} = w^l_{ij} * \boldsymbol{m}'_{ij}
$$

其中，顶点 $j \in \mathcal{N}(\boldsymbol{x}_i)$ 属于顶点 $i$ 的 $K$-近邻。此处我们设定 $K=30$。$\boldsymbol{h}^1_i = W_m \boldsymbol{h}_i$，其中 $W_m \in \mathbb{R}^{256 \times 2}$ 是一个映射矩阵，$l=1$ 到 $L_l$ 是局部视角建模模块的层数。$W^l_s \in \mathbb{R}^{1 \times 256}$ 和 $b^l_s \in \mathbb{R}$ 是可学习参数。$[;]$ 表示拼接操作。$\phi_e$ 表示多层感知机（MLP）。SiLU 表示 SiLU 激活函数。

对于每个顶点，我们从其邻居处传播消息以更新节点特征：

$$
\boldsymbol{c}^{l+1}_i = \sum_{j \in \mathcal{N}(\boldsymbol{x}_i)} \boldsymbol{m}^{l+1}_{ij}\\
\boldsymbol{h}^{l+1}_i = \boldsymbol{h}^l_i + \text{gate}(\boldsymbol{c}^{l+1}_i) \odot \boldsymbol{c}^{l+1}_i
$$

其中，gate 是通过一个 MLP 后接 sigmoid 函数实现的门控机制，用于控制信息流在局部几何形状上的传递。

### 全局景观建模

为了促进在整个期望表面上的消息传递，我们设计了一个全局景观编码器 FAMHA（图2(b) 右侧模块）。其核心思想是将帧平均技术（FA）（Puny 等，2021）融入一个多头注意力层。该操作不仅使全局生化特征得以扩散，还保证了其 SE(3) 等变性。

具体而言，从压缩后的点云 $\mathbf{X}' \in \mathbb{R}^{N' \times 3}$ 中，我们通过主成分分析（PCA）计算三个主成分向量 $v_1, v_2, v_3 \in \mathbb{R}^3$。利用这三个基本坐标，我们定义一个框架 $\mathcal{F}(\mathbf{X}')$ 为一个函数：

$$
\mathcal{F}(\mathbf{X}') = \{ ([\alpha_1 \boldsymbol{v}_1, \alpha_2 \boldsymbol{v}_2, \alpha_3 \boldsymbol{v}_3], t) | \alpha_i \in \{-1, +1\} \}
$$

其中，$t$ 是 $\mathbf{X}'$ 的质心。框架函数形成了一个包含八个变换的代数群。我们按如下方式计算全局消息传递：

$$
\widetilde{\boldsymbol{H}} = \frac{1}{|\mathcal{F}(\mathbf{X}')|} \sum_{g \in \mathcal{F}(\mathbf{X}')} \text{FAMHA}(\boldsymbol{H}^{l_l+1}; g^{-1} \mathbf{X}')
$$

其中，$\boldsymbol{H}^{l_l+1} = [\boldsymbol{h}^{l_l+1}_1, ..., \boldsymbol{h}^{l_l+1}_{N'}]^T \in \mathbb{R}^{N' \times 256}$ 是来自局部视角建模的输出顶点特征。$g^{-1} \mathbf{X}'$ 表示通过平移 $t$ 并使用旋转矩阵 $[\alpha_1 \boldsymbol{v}_1, \alpha_2 \boldsymbol{v}_2, \alpha_3 \boldsymbol{v}_3]$ 对 $\mathbf{X}'$ 进行旋转得到的结果，其中 $g \in \mathcal{F}(\mathbf{X}')$。FAMHA 由 $L_g$ 个堆叠的多头注意力（MHA）子层和全连接前馈网络（FFN）组成。在每个子层之后都执行残差连接和层归一化。因此，FAMHA 可以表述如下：

$$
\boldsymbol{h}^{l+1}_i = \text{LayerNorm}\left( \text{FFN}(\tilde{\boldsymbol{h}}^l_i) + \tilde{\boldsymbol{h}}^l_i \right),\\
\tilde{\boldsymbol{h}}^l_i = \text{LayerNorm}\left( \text{MHA}(\boldsymbol{h}^l_i, \boldsymbol{H}^l_g) + \boldsymbol{h}^l_i \right)
$$

其中，$l \in \{1, ..., L_g\}$ 且 $i \in \{1, ..., N'\}$。$h^l_i = [h^{l+1}_i; g^{-1} X']$，其中 $g \in \mathcal{F}(\mathbf{X}')$。$H^l_g = [h^l_1, ..., h^l_{N'}]^T$。

## 3.3 自回归蛋白质解码器

给定编码了几何形状和生化特征的隐藏表示，我们使用一个自回归 **Transformer 解码器**（即 GPT，Vaswani 等，2017）来为给定表面生成蛋白质序列：

$$
p(y_t) = \text{TransDec}(y_{<t}, \widetilde{\boldsymbol{H}}; \theta_{dec})
$$

其中，$p(y_t)$ 是蛋白质序列中第 $t$ 个残基的概率，$\theta_{dec}$ 表示可学习参数。

我们通过最小化负对数似然来训练整个模型：

$$
\mathcal{L} = \sum_{t=1}^{L} -\log p(y_t; \theta)
$$

其中，$\theta = \{\theta_{enc}, \theta_{dec}\}$ 表示我们的分层编码器和自回归蛋白质解码器的参数集合。

# 4. 实验

在本节中，我们首先在第 4.1 节描述我们的实现细节。然后，我们在一个通用的蛋白质设计任务——**逆折叠（Inverse Folding）**（第 4.2 节）和两个功能性蛋白质设计任务——**结合剂设计（Binder Design）**（第 4.3 节）与**酶设计（Enzyme Design）**（第 4.4 节）上，对所提出的 SurfPro 进行了广泛的实验评估。每个任务部分均介绍了具体的实验设置。

## 4.1 实现细节

我们为每个表面设定了 5,000 个顶点的最大限制。顶点数少于 5,000 的表面保持不变，而超过此限制的表面则通过下采样比率为 $r = 5,000 / N$ 进行压缩，其中 $N$ 表示原始顶点数。在表面压缩中，立方体内的最小顶点数 $N_{\min}$ 设定为 32。局部视角建模使用三层结构，全局景观建模采用两层 FAMHA。两个生化特征被映射到一个维度为 256 的隐藏空间。自回归解码器由一个 3 层 Transformer 解码器构建而成。小批量大小和学习率分别设定为 4,096 个 token 和 $5 \times 10^{-4}$。该模型在一块 NVIDIA RTX A6000 GPU 上训练，使用 Adam 优化器。生化特征的具体数值见附录表 9。

![image-20251110193834555](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110193834555.png)

## 4.2 逆折叠

本任务旨在设计能够折叠成给定主链结构的蛋白质序列。在我们的方法中，我们从一个更粗糙的结构——蛋白质表面——而非刚性的主链结构出发来设计蛋白质序列。

**数据集**：遵循先前的工作（Dauparas 等，2022；Gao 等，2022），我们使用由 Ingraham 等人（2019）整理的 CATH 4.2 数据集，并遵循 Jing 等人（2020）所使用的相同数据划分。由于 MSMS 工具在原始表面构建过程中偶尔会出现失败，以及蛋白质长度超过 1,024 个残基的情况，我们过滤掉了这些实例。因此，训练、验证和测试集分别包含 14,525、468 和 887 个样本。为了公平比较，我们对所有模型都严格使用相同的数据划分。经过筛选的 CATH 4.2 数据集的顶点数量统计见附录表 10。

![image-20251110193916493](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110193916493.png)

**基线模型**：我们将 SurfPro 与以下基线模型进行比较：(1) **ProteinMPNN** 是一种代表性的逆折叠模型。(2) **PiFold** 和 (3) **LM-DESIGN** 是逆折叠任务中的最先进方法。LM-DESIGN 所使用的架构是 LM-DESIGN（预训练的 ProteinMPNN-CMLM：微调）。我们使用它们在 GitHub 上发布的所有代码及其官方实现中的实验设置，以确保公平比较。

**评估指标**：遵循先前的工作（Jing 等，2020；Gao 等，2022），我们使用**困惑度（perplexity）**和**恢复率（recovery rate）**来评估所设计蛋白质序列的质量。由于表面不包含埋藏在下方的残基，我们在报告恢复率时会先进行成对序列比对，以确保公平比较：

$$
\text{recovery rate} = \frac{\text{number of recovered residue}}{\text{aligned sequence length}}
$$

我们在附录表 11 中提供了所有模型在成对比对后的恢复率，其中非自回归模型在比对后始终表现出低于比对前的恢复率，这与比对前的结果相比。

![image-20251110194001252](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110194001252.png)

**主要结果**：表 1 显示，在所有比较的基线中，SurfPro 达到了最高的恢复率和最低的困惑度。这些发现表明，将蛋白质表面的几何和生化约束结合起来，对于通用的蛋白质设计是有益的，从而使 SurfPro 在 CATH 4.2 数据集中实现了跨多种蛋白质折叠的最高恢复率。

![image-20251110194034301](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110194034301.png)

## 4.3 蛋白质结合剂设计

在本节中，我们的目标是使用 SurfPro 来设计能够以高亲和力与目标蛋白结合的蛋白质。

### 功能评估器

遵循先前的工作（Bennett 等，2023；Watson 等，2023），我们使用 AlphaFold2 (AF2) 的 pAE_interaction 来评估所设计结合剂与目标蛋白之间的结合亲和力。Bennett 等人（2023）发现，AF2 pAE_interaction 在区分实验验证的结合剂与非结合剂方面非常有效，其成功率范围在针对 IL7Ra、TrkA、InsulinR 和 PDGFR 等目标蛋白时为 1.5% 至 7%。我们使用 Bennett 等人（2023）提供的官方代码来计算 AF2 pAE_interaction。AF2 pAE_interaction 值越低，所设计的结合剂越好。

### 评估指标

我们使用贪心解码计算整个测试集的平均 AF2 pAE_interaction，并计算**平均成功率**。对于每个阳性结合剂，成功率被定义为具有更低 pAE_interaction 的所设计结合剂的比例。对于每一对 <阳性结合剂, 目标蛋白>，我们使用温度 $T=0.1$ 采样生成 10 个新的结合剂。为了计算 pAE_interaction，我们首先使用 ESMFold 预测所设计结合剂序列的结构，然后将该结构叠加到真实复合物上。最后，我们计算新复合物的 AF2 pAE_interaction。由于 AF2 pAE_interaction 模型会自动修正输入的复合物结构，因此 AlphaFold2 和 ESMFold 预测的结合剂结构之间几乎没有差异。相关比较见附录表 14。

![image-20251110194840994](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110194840994.png)

### 数据集

我们从 Bennett 等人（2023）收集了六个类别中经过实验验证的阳性 <结合剂, 目标蛋白> 复合物。在 10 个类别中，有 4 个类别的 AF2 pAE_interaction 在阴性和阳性结合剂之间无法区分。因此，我们选择在剩余的 6 个类别上进行评估，以确保评估的可靠性。对于包含超过 50 个复合物的类别，我们采用 8:1:1 的随机划分用于训练、验证和测试集；否则，所有复合物均被纳入测试集，建立一个零样本场景。详细的数据统计见附录表 12。

![image-20251110194916054](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110194916054.png)

### 基线模型

使用结合剂设计数据集，我们对所有基线模型（ProteinMPNN、PiFold、LM-DESIGN）以及我们的 SurfPro 进行微调，这些模型分别已在第 4.2 节详述的 CATH 4.2 数据集上进行了预训练。此外，我们还提供了一个随机基线，通过随机突变结合剂 20% 的残基得到。此外，为了充分利用 SurfPro 的全部设计能力，我们使用从整个蛋白质数据库（PDB）生成的所有表面对其进行预训练。这个预训练数据集截至 2023 年 3 月 10 日，包含 179,278 对 <表面, 序列>。详细的数据预处理步骤和预训练细节见附录 E。然后，我们在结合剂设计数据集上对该模型进行微调，我们将得到的模型称为 **SurfPro-Pretrain**。

### 主要结果

AF2 pAE interaction (↓)

![image-20251110194952405](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110194952405.png)

Success rate (%, ↑)

![image-20251110195103970](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110195103970.png)

不同模型的结果分别报告在表 2 和表 3 中。结果显示，我们的 SurfPro 在六个目标蛋白上实现了最低的平均 AF2 pAE_interaction 和最高的平均成功率。特别是，我们的 SurfPro 在所有六个类别上都取得了最佳的 pAE_interaction，并在其中三个类别上取得了最高的成功率。我们模型的 pAE_interaction 甚至略低于在 IL7Ra 上经过实验验证的阳性结合剂。这些发现证明，利用蛋白质表面特性对于功能性结合剂设计是有效的。此外，SurfPro 在两个零样本测试类别中实现了最高的成功率，证实了其直接从表面捕获有价值蛋白质特性的能力。因此，即使没有针对特定目标蛋白结合进行专门训练，SurfPro 也能在那些 pAE_interaction 低于阳性结合剂的类别中生成结合剂。在对整个 PDB 进行预训练后，通过贪心解码生成的结合剂功能显示出微小差异。然而，成功率显著提高，从 22.29% 提升至 26.22%。这表明，在更大规模的数据集上进行预训练有助于提升 SurfPro 的设计能力，确保能设计出更多具有更好 pAE_interaction 的结合剂。

## 4.4 酶设计

在我们的工作中，我们的目标是设计能够与特定底物结合的酶。

### 功能评估器

为了评估酶与底物之间的结合亲和力，我们使用 Kroll 等人（2023a）开发的 ESP 评分。他们的模型预测酶-底物相互作用的准确率高达 91%。我们使用他们的官方代码来计算 ESP 评分。

### 评估指标

与结合剂设计类似，我们报告使用贪心解码的平均 ESP 评分，以及使用温度 $T=0.1$ 采样得到的平均成功率。

### 数据集

我们从 Kroll 等人（2023a）收集了五个类别的酶，每个类别都针对一种特定的底物。我们排除了 CATH 4.2 中的酶，以防止数据泄露问题。对于包含超过 100 个样本的酶类别，我们在聚类后按 8:1:1 的比例随机划分数据为训练、验证和测试集；否则，所有数据均作为测试集。详细的数据统计见附录表 13。

![image-20251110200247089](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110200247089.png)

### 基线模型

与结合剂设计类似，我们使用酶设计数据集对所有基线模型（ProteinMPNN、PiFold、LM-DESIGN）以及我们的 SurfPro 进行微调，这些模型分别基于逆折叠任务中预训练的模型。

同样地，我们也提供了在相同设置下（如结合剂设计）的随机基线和 SurfPro-Pretrain 的结果。请注意，预训练数据集在此处排除了所有酶，以防止数据泄露问题。

### 主要结果

ESP score (↑)

![image-20251110200329483](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110200329483.png)

Success rate (%, ↑)

![image-20251110200405951](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110200405951.png)

表 4 和表 5 显示，我们的 SurfPro 在五个类别中实现了最高的平均成功率，并且其平均 ESP 评分与 LM-DESIGN 相当。需要注意的是，LM-DESIGN 是在 650M ESM-1b (Rives 等, 2021) 上进行微调的，而后者又是在庞大的 UniRef50 数据集上预训练的。因此，存在数据泄露的可能性，这使其能够在平均 ESP 评分上取得最佳表现。然而，我们的 SurfPro 在成功率方面显著优于 LM-DESIGN，达到了 42.23%，而 LM-DESIGN 为 37.58%。

这一性能在对整个 PDB 表面进行预训练后进一步提升至 43.63%。这些发现表明，我们的 SurfPro 能够设计出比天然酶具有更强酶-底物相互作用功能的酶，再次验证了表面特性在功能性蛋白质设计中的帮助作用。此外，我们的 SurfPro 展示了零样本设计能力，在设计能与底物 C00001 结合的酶时，成功率达到 33.55%。

# 5. 分析：深入探究 SurfPro

---

## 5.1 消融研究：各组件如何发挥作用？

几何和生化约束均有助于蛋白质设计。为了更好地分析我们模型中不同组件的影响，我们在逆折叠任务上进行了消融测试。待比较的模型如下：(1) **SurfPro-w-five** 使用五种生化特征，即疏水性、电荷、极性、受体和供体；(2) **SurfPro-w/o-global** 移除了全局景观建模；(3) **SurfPro-w/o-local** 移除了局部视角建模；(4) **SurfPro-w-hydrophobicity** 仅使用疏水性特征；(5) **SurfPro-w-charge** 仅使用电荷特征；(6) **SurfPro-w/o-feature** 不使用任何生化特征；(7) **SurfPro-w-unsorted** 不对原始表面上的顶点进行排序。

![image-20251110201042600](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110201042600.png)

表6的结果表明，引入五种化学特征并未带来额外收益。移除全局景观建模或局部视角建模会导致性能显著下降。仅使用疏水性特征会轻微降低性能，而仅依赖电荷特征则会严重损害性能。缺少所有生化特征会进一步降低性能。这些观察结果验证了在表面表示学习中，几何形状和生化特征都扮演着至关重要的角色，强调了将两者都纳入蛋白质设计过程的必要性。值得注意的是，不对原始表面上的顶点进行排序会显著降低性能。我们的解释是，不同区域的局部形状可能相似，而模型在不排序顶点的情况下很难将每个局部形状与特定的蛋白质片段对齐，尤其是在序列极长时。

## 5.2 消融研究：自回归解码器能否被替代？

自回归解码器在我们提出的 SurfPro 架构中表现出强大的性能。为了评估其重要性，我们将 SurfPro 与一个采用非自回归解码器的替代方案进行比较，该方案融合了受自然语言处理中非自回归机器翻译启发的 SoftCopy 和扫视（glancing）学习策略（Gu & Kong, 2021; Qian 等, 2021）。表7展示了比较结果。这明确证明了 SurfPro 的表现优于非自回归解码器变体，证实了自回归解码器在我们所提出框架中的有效性。

![image-20251110201122369](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110201122369.png)

## 5.3 SurfPro 能否设计新颖且多样的蛋白质？

![image-20251110201216404](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110201216404.png)

SurfPro 能够生成新颖且多样的蛋白质。在温度为 0.1 的条件下采样得到的所设计酶的新颖性分布如图3(a)所示。此处的新颖性计算方式为 1 - 恢复率。该图显示，我们的 SurfPro 表现出优于 ProteinMPNN 的平均新颖性（58.51% 对比 49.46%）。此外，我们还分析了不同位置上所设计结合剂的氨基酸分布。图4以目标蛋白 InsulinR 为例，展示了一个多样化的残基分布。值得注意的是，所有这些所设计的结合剂（附录F.2中提供）都实现了低于10的 pAE_interaction，这是一个由 Bennett 等人 (2023) 确定的阈值，用以显著提升成功率。这些发现证实了 SurfPro 能够设计出具有所需功能的多样化蛋白质。

![image-20251110201247217](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110201247217.png)

## 5.4 顶点数量如何影响蛋白质设计？

采样的顶点越多，SurfPro 的表现越有效。为了彻底探索表面顶点采样规模对模型性能的影响，我们在 CATH 4.2 数据集上训练了最大顶点数从 1k 到 10k 的模型。图3(b) 展示了随着采样顶点数的增加，恢复率得到了提高。然而，在采样规模超过 5k 后，提升速率略有下降。此外，随着采样顶点数的增加，推理速度会减慢。为了确保设计质量和推理效率，我们将最大顶点数设定为 5k。

## 5.5 与 MaSIF 的比较

![image-20251110201339784](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110201339784.png)

在本节中，我们对我们的 SurfPro 与成熟的基于表面的结合剂设计模型 MaSIF (Gainza 等, 2020) 进行了对比分析。首先，我们使用 ProteinMPNN 为每个阳性结合剂生成 100 条结合剂序列。随后，利用 ESMFold 预测这些所设计结合剂序列的结构。在此之后，MaSIF 被用于对这些候选结合剂进行排序，并选择表现最佳的一个作为最终设计的结合剂。在六个数据集上的最终 pAE_interaction 得分总结于表8。比较结果显示，MaSIF 在 6 个目标蛋白中的 2 个上优于我们的模型，而我们的 SurfPro 在 4 个上表现更优。

尽管 MaSIF 展示了略高的平均 pAE_interaction 得分，但值得注意的是，MaSIF 识别一个有前景的结合剂所需的平均时间为 210.42 秒，远长于 SurfPro 所需的 0.45 秒。这突显了 SurfPro 作为一个独立生成模型的有效性，它能够快速、直接地以端到端的方式生成功能性蛋白质序列。

## 5.6 案例研究

![image-20251110201410919](./SurfPro%20Functional%20Protein%20Design%20Based%20on%20Continuous%20Surface.assets/image-20251110201410919.png)

为了深入了解我们 SurfPro 设计的功能性蛋白质，我们可视化了两个属于 TrkA（图5(a)）和 PDGFR（图5(b)）的目标蛋白的模型设计结合剂复合物。这两个复合物的 AF2 pAE_interaction 均低于 6，表明存在强蛋白-蛋白结合。正如 Bennett 等人 (2023) 在他们的工作中所指出的，当设计的 AF2 pAE_interaction < 10 时，成功率将显著提高。直观来看，这表明我们的 SurfPro 能够设计出具有高蛋白-蛋白结合亲和力的功能性结合剂。

# 6. 讨论

我们的 SurfPro 在基于给定蛋白质表面快速、直接地生成功能性蛋白质序列方面表现出卓越的性能。然而，尽管其能力令人印象深刻，但仍存在一些局限性，我们将在本节中进行讨论。虽然我们的方法在蛋白质优化方面表现出色，但它更倾向于精修（refinement）而非从头设计（de novo protein design）。这一区别具有重要意义。特别是在结合剂设计中，从零开始设计一个高亲和力的结合剂在实际场景中很少可行。因此，从现有的阳性结合剂出发可以加速设计过程。然而，这种方法也对我们的方法的实际应用施加了限制。首先，定位一个合适的初始点并不总是可行的。其次，从一个有利的起点开始可能导致改进有限，因为我们的 SurfPro 需要考虑几何和生化约束。

我们未来的工作可以探索用于从头设计蛋白质表面的方法。例如，整合扩散模型来生成点云，可以显著增强我们现有框架的通用性和适用性。

# 7. 结论

在本工作中，我们提出了 SurfPro，这是一种基于期望表面设计功能性蛋白质的新生成模型。SurfPro 包含一个分层编码器，该编码器逐步捕捉几何和生化特征，从局部视角过渡到全局景观。此外，还采用了一个自回归解码器，根据学习到的表面几何和生化表示来生成蛋白质序列。我们的方法在通用的蛋白质设计基准 CATH 4.2 上持续优于先前强大的逆折叠方法，序列恢复率达到 57.78%；并在两个功能性蛋白质设计任务中，在蛋白质-蛋白质结合和酶-底物相互作用得分方面取得了更高的成功率。