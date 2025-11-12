# Denoising diffusion probabilistic models on SO(3) for rotational alignment.

# 摘要

概率扩散模型能够为一系列应用在高维欧几里得空间上对复杂的数据分布进行建模。然而，许多现实世界的任务涉及更复杂的结构，例如流形上定义的数据分布，这些数据分布无法通过 Rn 上的扩散轻松表示。本文针对涉及 3D 旋转的任务提出了利用李群 SO(3) 上的扩散过程的去噪扩散模型，以便生成旋转对齐任务的候选解决方案。实验结果表明，在合成旋转分布采样和 3D 对象对齐任务中，所提出的 SO(3) 扩散过程优于欧拉角扩散等简单方法。

# 背景

去噪扩散概率模型（DDPMs）能够从复杂分布中生成高质量的样本，在音频合成和图像应用中取得了令人鼓舞的成果。然而，存在许多问题（例如姿态估计和蛋白质对接），其领域 $\mathbb{R}^n$ 并不适用。由于这些问题在本质上具有旋转-平移不变性，因此从一个条件扩散模型中对三维旋转群 $SO(3)$ 上的可能姿态进行采样是一种更合适的概率模型。

在本工作中，我们引入了在李群 $SO(3)$ 上的去噪扩散模型。

去噪扩散概率模型（DDPMs）是一类受非平衡热力学启发的生成模型。其核心思想是模拟一个扩散过程，该过程将某种形式的观测数据（例如图像），记为 $\mathbf{x}_0$（其分布未知，记为 $q(\mathbf{x}_0)$），转化为纯噪声。然后，可以通过学习逆向过程来找到一个生成模型，将噪声重新变回底层数据的结构。

在实践中，扩散过程被替换为一个非齐次离散时间马尔可夫链，其一步转移密度为：

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

其中，$\beta_t, t=1, ..., T$ 表示方差调度表，$\mathcal{N}(y; \mu, \Sigma)$ 表示均值为 $\mu$、协方差矩阵为 $\Sigma$ 的高斯密度。在适当条件下，最终值 $\mathbf{x}_T$ 将近似服从高斯分布 $q(\mathbf{x}_T) \approx \mathcal{N}(0, \mathbf{I})$。

去噪模型学习的是逆向过程 $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 的近似，其中 $p(x_T) = \mathcal{N}(0, \mathbf{I})$。因此，转移核 $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 学习预测前向过程的上一时间步，并由一个正态分布参数化：

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t)).
$$

函数 $\mu_\theta$ 和 $\Sigma_\theta$ 是具有可学习参数 $\theta$ 的神经网络的输出。最近的工作表明，在公式 (2) 中将协方差矩阵 $\Sigma_t$ 固定可以带来更好的性能。对前向过程 $x_t \sim q(\mathbf{x}_t | \mathbf{x}_0) = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + (1 - \bar{\alpha}_t) \epsilon$ 进行进一步重参数化，其中 $\epsilon \sim \mathcal{N}(0, I)$，$\alpha_t = 1 - \beta_t$ 且 $\bar{\alpha}_t = \prod_{s=0}^{t} \alpha_s$，结果如下：

$$
\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right).
$$

损失方程随后可以简化为关于所添加噪声的一个函数：

$$
L_t(\theta) = \mathbb{E}_{\tau, \epsilon, \mathbf{x}_0} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_\tau, \tau) \|^2 \right].
$$

### 2 在旋转群 $SO(3)$ 上定义扩散

训练扩散模型的一个关键组成部分是在时间步 $t$ 时从扩散分布中采样，而无需计算中间值。对于 $\mathbb{R}^n$（欧几里得空间）上的正态分布扩散，这可以通过之前推导的闭式方程轻松实现。然而，由于多种因素，这很难推广到旋转空间 $SO(3)$。

1. 直接在欧几里得空间扩散的局限性
   - **欧拉角扩散问题**：
     - 使用欧拉角 $(\psi, \vartheta, \phi)$ 将旋转表示为 $\mathbb{R}^3$ 上的扩散，会破坏旋转空间的对称性，因此无法正确反映真实旋转的几何性质。
   - **四元数扩散问题**：
     - 虽然四元数提供了旋转的连续表示，但必须保持单位长度（位于三维球面 $S^3$ 上），导致每一步都要重新归一化。
     - 这种非线性投影使得传统扩散采样的闭式更新（如 Ho et al., 2020 中的快速采样）失效。
     - 同时，四元数是 $SO(3)$ 的**双重覆盖**（即每个旋转对应两个符号相反的四元数），这种表示冗余对神经网络训练不友好。
2. 需要在 $SO(3)$ 上定义合理的“高斯分布”：目标是找到一种在 $SO(3)$ 上定义的分布，它应当具备类似正态分布的性质，特别是：
   - 在卷积下封闭（即分布的组合仍然是同类型的分布）；
   - 能够方便地推导前向扩散方程 $q(\mathbf{x}_t | \mathbf{x}_0)$；
   - 能实现高效采样。
3. 现有旋转分布的不足
   - **Fisher 分布**（用于旋转矩阵）和 **Bingham 分布**（用于单位四元数）虽然都能描述旋转不确定性， 但它们：
     - **不在卷积下封闭**；
     - 导致在任意时间步计算扩散分布（如 $q(R_t|R_0)$）变得复杂；
     - 无法实现高效的直接采样。

相反，我们考虑 $SO(3)$ 上的各向同性高斯分布 (IG) $g \sim \mathcal{IG}_{SO(3)}(\mu, \epsilon^2)$，它由一个平均旋转 $\mu$ 和标量方差 $\epsilon$ 参数化。IG 分布可以用轴角形式参数化，其轴均匀采样，旋转角 $\omega \in [0, \pi]$ 的密度为：

$$
f(\omega) = \frac{1 - \cos \omega}{\pi} \sum_{l=0}^{\infty} (2l+1) e^{-l(l+1)\epsilon^2} \frac{\sin((l+\frac{1}{2})\omega)}{\sin(\omega/2)}.
$$

请注意，$SO(3)$ 上的均匀分布，记为 $U_{SO(3)}$，其参数化为均匀轴和 $f(\omega) = \frac{1 - \cos \omega}{\pi}$，在从该分布采样时需要作为缩放因子包含在内。

IG 分布既是中心极限定理 (CLT) 的自然扩展，也是 $SO(3)$ 上布朗运动期望分布的扩展，这为其用于去噪扩散模型提供了强有力的动机。虽然 IG 分布的方差被定义为标量值，因此相比 Matrix-Fisher 或 Bingham 分布灵活性较低，并且欧氏去噪扩散模型假设维度之间没有相关性，因而也以标量方差参数化。最重要的是，IG 分布在卷积下是封闭的。

我们可以利用这种关系推导出与欧几里得扩散过程类似的方程，使我们能够在任意时间步定义分布，从而实现高效采样。最初，数据 $\mathbf{x}_0 \in SO(3)$ 从分布 $q(\mathbf{x}_0)$ 中采样并进行扩散。如前所述，在 $t=T$ 时，我们认为数据已完全扩散。

为了推导分布 $q(\mathbf{x}_t | \mathbf{x}_0)$，请注意，类似的欧几里得分布（公式 1）需要对 $\mathbf{x}_0$ 项应用一个缩放项，因为结果值不位于 $SO(3)$ 流形上。直接应用于旋转矩阵的缩放项是没有意义的，因为它可以被视为远离原点的平移。如果 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ 被视为远离恒等旋转 $I$ 的旋转，这意味着 $\mathbf{x}_0 \sim s(\mathbf{x}_0)$ 可以被视为远离恒等旋转 $I$ 的旋转，我们可以沿着从恒等旋转出发的测地线通过插值旋转角度来缩放我们的旋转。为此，我们依赖于李代数 $\mathfrak{so}(3)$ 和 $SO(3)$ 之间的指数映射和对数映射。直观地说，一个旋转矩阵 $R$ 具有一个关联的旋转角 $\theta$，而旋转矩阵 $P = RR = R^2$ 具有 $2\theta$ 的旋转角。我们遵循标准定义 (Cardoso & Leite, 2010) 来定义旋转矩阵的对数，并将其定义为：$\log R = \frac{\theta}{2 \sin \theta} (R^\top - R)$

其中 $\theta$ 满足 $1 + 2 \cos \theta = \text{trace}(R)$。$\mathfrak{so}(3)$ 中的矩阵是斜对称的，形式为 $S(v)$：$S(v) = \begin{pmatrix}
0 & z & -y \\
-z & 0 & x \\
y & -x & 0
\end{pmatrix}, \quad v = [x, y, z],$

其中 $\|v\|_2 = \theta$。根据旋转矩阵对数的这一定义，我们能够通过将它们转换为李代数 $\mathfrak{so}(3)$ 中的值、逐元素乘以标量值，再通过矩阵指数运算转换回旋转矩阵来缩放旋转矩阵。旋转的组合通过 $SO(3)$ 中的矩阵乘法完成，类似于欧几里得扩散模型中的加法。

$$
\lambda(\gamma, \mathbf{x}) = \exp(\gamma \log(\mathbf{x})).
$$

因此，函数 $\lambda(...)$ 是从 $I$ 到 $x$ 的测地流，其大小为 $\gamma$。将这些应用于原始 DDPM 模型的方程，我们得到以下定义：

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{IG}_{SO(3)}(\lambda(\sqrt{\bar{\alpha}_t}, \mathbf{x}_0), (1 - \bar{\alpha}_t));
$$

$$
p(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{IG}_{SO(3)}(\tilde{\mu}(\mathbf{x}_t \mathbf{x}_0), \tilde{\beta}_t)
$$

以及

$$
\tilde{\mu}(\mathbf{x}_t, \mathbf{x}_0) = \lambda\left( \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t}, \mathbf{x}_0 \right) \lambda\left( \frac{\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}, \mathbf{x}_t \right).
$$