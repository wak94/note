# PROTPAINTER: DRAW OR DRAG PROTEIN VIA  TOPOLOGY-GUIDED DIFFUSION

# 摘要

近期在蛋白质主链生成方面的进展，在结构、功能或物理约束下已取得了令人瞩目的成果。然而，现有方法缺乏对拓扑结构进行精确控制的灵活性，从而限制了对主链空间的探索。我们提出了 **ProtPainter**，这是一种基于扩散的方法，用于根据 3D 曲线生成蛋白质主链。

ProtPainter 遵循一个两阶段流程：曲线草图绘制和草图引导的主链生成。在第一阶段，我们提出 **CurveEncoder**，它从曲线预测二级结构注释，以参数化草图的生成。在第二阶段，该草图指导去噪扩散概率模型（DDPM）的生成过程来生成主链。在此过程中，我们进一步引入了一种融合调度方案——**Helix-Gating**——来控制缩放因子。

为了评估，我们提出了首个针对拓扑约束蛋白质生成的基准测试，并引入了“蛋白质恢复任务”和一种新的度量标准——“自洽拓扑适应性”（scTF）。实验结果证明，ProtPainter 能够生成拓扑适配性高（scTF > 0.8）且可设计性强（scTM > 0.5）的主链，其绘图与拖拽任务也展示了其灵活性与通用性。

# 1. 背景

- **去噪扩散模型**（如 RFdiffusion, Genie, FrameDiff）和**基于流的模型**（如 FrameFlow, FOLDFLOW-OT）已能生成逼真的蛋白质主链结构。
- 然而，**如何引导生成过程以满足设计者的结构或功能意图**仍是挑战。
- 现有方法仅能利用生化属性、功能、接触图、结构基序或对称性等条件进行约束，但缺乏**精细且可编辑的结构控制机制**。
- **拓扑结构（CG Topo）**提供了更高层次的结构抽象，但目前依赖参数化配置，限制了三维拓扑的多样性和灵活性。

## 核心思想

- **ProtPainter** 首次提出以 **3D 曲线（3D curves）** 作为拓扑约束，定义蛋白质折叠过程。
- 3D 曲线能够自然描述蛋白质的**螺旋数量、长度、方向、曲率与空间分布**，提供更精细的拓扑控制。
- 目标类似图像超分辨率任务：
  - 粗糙的 3D 曲线 → 细化为完整蛋白质主链。
  - 曲线条件对应“参考图像”，主链生成对应“高分辨率图像”。

## 方法设计

- 提出 **CurveEncoder** 模块：
  - 将生成的主链下采样为“frame”（框架）。
  - 将条件曲线上采样为“sketch”（草图）。
  - 使两者共享相同维度，便于在 DDP M 框架中进行条件生成。
- 借鉴 **RFdiffusion** 的引导机制，采用 **基于 RoseTTAFold 的自调节平移引导** 提升生成质量。
- 引入 **Helix-Gating** 融合调度机制，动态调节曲线引导强度。

## 主要贡献

1. **提出 ProtPainter 框架**：
   - 首个基于 3D 曲线生成具特定拓扑蛋白质主链的方法。
   - 包括 CurveEncoder（特征提取与SSE预测）与 Helix-Gating（曲线引导机制）。
2. **建立评测基准**：
   - 新指标 **scTF**：评估生成主链与曲线拓扑一致性。
   - 数据集与曲线操作工具（连接、拖拽、绘制）。
   - “蛋白质恢复任务”用于评测生成效果。
3. **验证下游应用潜力**：
   - 适用于结合剂设计、基序支架和多态蛋白（如铰链蛋白）的初步支架生成。
   - 证明该方法能实现更直观、灵活的蛋白质拓扑控制。

# 相关方法

## 2.1 扩散概率建模

扩散概率建模将模型训练表述如下：给定一个前向扩散过程，模型预测在时间 $t$ 添加到原始样本中的噪声。对于来自训练集的样本 $x_0$，前向过程被定义为迭代地在 $T$ 步内向样本添加少量高斯噪声，从而产生一个噪声样本序列 $x_{0:T}$，使得最终样本 $x_T \sim \mathcal{N}(0, 1)$ 是一个很好的近似。

在去噪扩散概率建模（DDPM）（Ho 等，2020）的框架下，每一步的噪声幅度由一个方差调度表 $\beta_t, t \in [0:T]$ 定义，满足：

$$
p_t(x_t | x_{t-1}) = \mathcal{N}(x_t, \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)   
$$

上述转换定义了一个马尔可夫过程，其中原始数据被转化为标准正态分布。可以将给定 $x_0$ 的 $x_t$ 的密度写成闭式形式：

$$
p_t(x_t | x_0) = \mathcal{N}(x_t, \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I), \quad \text{s.t.} \quad x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t,
$$

其中 $\bar{\alpha}_t = \Pi_{i=1}^t \alpha_i$ 且 $\alpha_i = 1 - \beta_i$，而 $\epsilon_t \sim \mathcal{N}(0, 1)$。

将样本 $x_T$ 转换回样本 $x_0$ 是通过几个更新步骤完成的，这些步骤逆转了添加破坏性噪声的过程，由一个反向采样方案给出：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\sqrt{1 - \alpha_t}}{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t) \right) + (1 - \alpha_t) \epsilon,
$$

其中 $\epsilon \sim \mathcal{N}(0, 1)$。神经网络 $\epsilon_\theta$（即去噪器）被训练以预测添加到 $x_0$ 的噪声。

## 2.2 用于主链生成的引导扩散

引导采样已应用于扩散模型中，以根据人类指令生成样本，其采样过程被表述为条件项和无条件项。考虑到条件变量 $y$，像 Chroma 这样的分类器引导方法会在噪声结构 $x_t \sim p_t(x|x_0)$ 上训练一个时间依赖的分类器模型 $p_t(y|x)$，并通过贝叶斯推断调整采样后验 $\nabla_x \log p_t(x|y)$。与训练分类器来估计 $p_t(y|x)$ 不同，无分类器方法则启发式地近似条件项。Wang 等人（2024）和 Komorowska 等人（2024）引入了物理力，以将无条件模型扩展到动态构象采样。力引导也被应用于生成能量更低的抗体。

## 2.3 基于拓扑的主链控制

控制蛋白质拓扑结构一直是超越 SSE 线性配置（也称为“蓝图”）的一个长期挑战。常见的基于蓝图的方法参数化地描述二级结构元件的数量、大致位置以及整体朝向，这对于从头蛋白质设计、基序支架设计和膜蛋白设计是初步但至关重要的。这些参数化表示在重复组装方面很强大，但由于其自由度有限，使得普通用户难以灵活地控制更复杂的三维拓扑结构。

### 用于主链生成的基于拓扑的扩散模型

为了获得比传统蓝图方法更多的控制，一些基于扩散的模型已将拓扑结构作为条件：Topodiff 利用 VAE 在潜在空间中编码全局拓扑，从而通过提供查询蛋白实现拓扑控制，但不支持更详细的拓扑编辑。DiffTopo 利用扩散模型从预定义的 SSE 序列生成草图，然后将其输入 RFdiffusion 以生成逼真的主链。然而，它的拓扑定义遵循线性配置，缺乏对三维拓扑的控制。总之，当前的基于拓扑的扩散模型不支持更灵活和更详细的拓扑控制。

# 3. 方法

![image-20251110103555195](./PROTPAINTER%20DRAW%20OR%20DRAG%20PROTEIN%20VIA%20%20TOPOLOGY-GUIDED%20DIFFUSION.assets/image-20251110103555195.png)

## 3.1 绘制曲线

本阶段首先通过定义一个前向过程，将主链拓扑结构抽象为曲线表示。接着，我们引入 **CurveEncoder**，它使用 SSE 标签对曲线进行注释，记为 SSEcurve。最终的草图则在 SSEcurve 的指导下参数化生成。

### 由曲线表示的拓扑结构

为了用曲线来表示 $C_\alpha$ 主链的拓扑结构，我们采用附录 A.1 中详述的下采样方法。具体而言，α-螺旋和 β-折叠被抽象为其各自的中心轴，而环区则保留其原始坐标。然后，所得的曲线坐标会经过重新采样、平滑处理，并通过对其最近邻主链原子的标签取平均值来标注 SSE 标签。

### CurveEncoder

该模块旨在预测曲线 SSEcurve 的 SSE 注释，作为曲线 SSE 分配的逆向过程。受 Greener & Jamali (2022) 的启发，我们应用一个三层 EGNN（Satorras 等，2021）来提取曲线坐标的连接性特征，并使用一维 CNN 来提取曲率特征作为补充。随后，一个多头注意力层整合这些特征，并预测 SSEcurve 的二级结构元件注释。SSEcurve 可以根据用户输入进行自定义。给定第 $l$ 层的节点嵌入 $h^l$ 和坐标嵌入 $x^l$，以及边信息 $\varepsilon = (e_{ij})$，等变图卷积层 (EGCL) 的表达式为 $h_{l+1}, x_{l+1} = \text{EGCL}[h_l, x_l, \varepsilon]$。算法如算法 1 所示，更多细节见附录 D。使用曲线坐标和 SSEcurve 生成朴素草图的参数化方法见附录 E。

## 3.2 草图引导的主链采样

我们的模型采用 Watson 等人（2023）提出的框架表示法，该表示法包含每个残基的平移 $z$（$C_\alpha$ 坐标）和旋转 $r$（$N-C_\alpha-C$ 刚性取向）。设 $X_t = [z_t, r_t]$ 为扩散步骤 $t$ 时的残基框架，其中 $z_t \in \mathbb{R}^{N_{res} \times 3}$ 是 $C_\alpha$ 的坐标（平移部分），而 $r_t \in SO(3)^{N_{res}}$ 是旋转矩阵（旋转部分）。

平移部分由 DDPM 中的三维高斯噪声生成：

$$
p(z_{t-1} | z_t, z_0) = \mathcal{N}(z_{t-1}; \tilde{\mu}(z_t, z_0), \tilde{\beta}_t I_3), \quad \text{with} \quad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \approx \beta_t,
$$

其中，

$$
\tilde{\mu}(z_t, z_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} z_0 + \frac{\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} z_t,
$$

且 $z_0$ 可以通过 RoseTTAFold 对带有掩码序列输入的预测 $\hat{z}_0$ 来估计，灵感来源于 RFdiffusion。

对于残基取向，我们在旋转矩阵流形上使用布朗运动。这些框架在旋转下是等变的：

$$
p_\theta(x_{t-1} | x_t) = p_\theta(R * x_{t-1} | R * x_t), \quad \text{where} \quad R * x_t = [Rz, Rr].
$$

给定一个参考朴素草图 $y \in \mathbb{R}^{N_{res} \times 3}$，我们定义一个条件分布作为引导项 $y_{t-1}$：

$$
y_{t-1} \sim q(y_{t-1} | y, \hat{z}_0^t).
$$

Choi 等人（2021）通过参考图像 $y$ 来精炼生成的 $x$，其条件分布可近似为：

$$
p_\theta(x_{t-1} | x_t, y) \approx p_\theta(x_{t-1} | x_t, \phi(x_{t-1}) = \phi(y_{t-1})).
$$

其中，$\phi$ 是一个线性低通滤波操作，以确保参考图像的低频特征与生成图像保持一致。我们采用了这一思路，将低维的生成结果与草图对齐，草图在此过程中充当桥梁。所提出的 CurveEncoder 将条件曲线向上采样为一个草图，而 $\phi$ 则将生成的主链向下采样为一个框架。生成的主链框架 $x_0$ 现在可以通过参考草图 $y$ 和经过滤波的框架部分 $\phi_\lambda(x_0)$ 进行引导。我们定义框架滤波操作为 $\phi_\lambda(X_t) = \lambda z(X_t)$，其中 $z(X_t)$ 提取框架 $X_t$ 的 $C_\alpha$ 坐标，而 $\lambda$（介于 0 和 1 之间）是一个用于权衡多样性和引导性的因子。

我们近似地将旋转 ($r_t$) 和平移 ($z_t$) 视为在一般平移条件 $c_T$ 下独立分布的变量：

$$
p_\theta(x_{t-1} | x_t, c_T) = p_\theta(z_{t-1} | x_t, c_T) p_\theta(r_{t-1} | x_t) \quad \text{if} \quad p_\theta(r_{t-1} | x_t) = p_\theta(r_{t-1} | x_t, c_T).
$$

结合公式 (8) 和 (9)，我们有：

$$
p_\theta(x_{t-1} | x_t, y) \approx p_\theta(z_{t-1} | x_t, \phi(x_{t-1}) = \phi(y_{t-1})) p_\theta(r_{t-1} | x_t).
$$

我们只需更新 $z_{t-1}$：

$$
z_{t-1} = \phi(y_{t-1}) + I z'_{t-1} - \phi(x'_{t-1})
$$

其中，$x'_{t-1}$ 从由 $x_t$ 提出的无条件分布中采样，即 $x'_{t-1} \sim p_\theta(x'_{t-1} | x_t)$，而 $z'_{t-1}$ 是 $x'_{t-1}$ 的平移部分。将操作 $\phi$ 应用于公式 (5)，我们得到：

$$
\phi(z'_{t-1}) = \frac{\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \phi(z_t).
$$

因此，条件概率近似为：

$$
z_{t-1} = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \cdot \hat{z}_0^t + \frac{\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot (1 - \lambda) \cdot z'_t + \lambda \cdot y_{t-1}.
$$

## 3.3 Helix-Gating：控制缩放因子

![image-20251110150202152](./PROTPAINTER%20DRAW%20OR%20DRAG%20PROTEIN%20VIA%20%20TOPOLOGY-GUIDED%20DIFFUSION.assets/image-20251110150202152.png)

我们提出了 **Helix-Gating**，这是一种两阶段融合调度方案，旨在增强草图 $y$ 引导的采样过程。在“保密阶段”和“可控阶段”之间的过渡时机，是通过比较预测蛋白 $\hat{z}_0^t$ 和草图 $y$ 之间的螺旋百分比（记作算子 $\mathcal{O}$）来确定的。在保密阶段，引导作用受到限制并被缩放，利用预测蛋白与目标蛋白之间螺旋百分比的差异，确保在有限的草图引导下实现恒定的保真度提升。在可控阶段，引导作用被完全提供：

$$
y_{t-1} = \frac{\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \mathcal{F}(y, \hat{z}_0^t) \cdot y \quad (14)
$$

$$
\mathcal{F}(y, \hat{z}_0^t) =
\begin{cases}
\gamma \cdot \delta(\mathcal{O}(\hat{z}_0^t), \mathcal{O}(y)) + \eta, & \text{if } \mathcal{O}(\hat{z}_0^t) < \mathcal{O}(y) \quad \text{(Confidential Phase)} \\
I, & \text{if } \mathcal{O}(\hat{z}_0^t) \geq \mathcal{O}(y) \quad \text{(Controllable Phase)}
\end{cases}
$$

其中，$\gamma$ 和 $\eta$ 是超参数，$\delta$ 是差分函数。根据消融研究 G.5，我们设定 $\lambda = 2/3$、$\gamma = 0.2$、$\eta = 0.7$。图 2b 说明了此过程：一旦螺旋百分比达到阈值，可控阶段便开始，更多条件信息被整合进来，从而精细调整生成结果以更紧密地贴合草图。消融研究 5 和附录 12.c 证明了 Helix-Gating 的有效性。

## 3.4 绘制结合剂与拖拽蛋白

拖拽一个蛋白质被表述为基于曲线的基序支架化。在此过程中，被拖拽蛋白的曲线充当待生成的支架，其更新后的形状作为基序条件。对于长度为 $L$ 的结构，令 $M$ 和 $S$ 分别为基序和支架的索引集，即 $M \cup S = \{1, ..., L\}$。因此，设基序和支架的结构分别为 $x^M$ 和 $x^S$。整个无噪声结构为 $x_0 = [x_0^M, x_0^S]$。$x_0^M$ 主链和侧链结构作为 RoseTTAFold 的固定模板来预测 $\hat{z}_0^S$，这会影响平移部分；当基序被固定或掩码时，去噪过程变为 $p_\theta(x_{t-1}^S | x_t^S, x_0^M, c_T)$。我们近似地认为：

$$
p_\theta(x_{t-1}^S | x_t^S, x_0^M, c_T) \approx p_\theta(z_{t-1}^S | x_t^S, c_T) p_\theta(r_{t-1}^S | x_t^S, x_0^M)
$$

为了绘制结合剂，我们首先计算曲线与目标蛋白之间的距离，以识别目标上的“界面热点” $z_h$。然后，复杂的结构设计同时以曲线和目标蛋白为条件，将这些热点残基纳入考虑。