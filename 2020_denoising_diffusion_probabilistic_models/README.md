# Denoising Diffusion Probabilistic Models (DDPM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanicey/notebook/blob/main/2020_denoising_diffusion_probabilistic_models/notebook.ipynb) &nbsp; [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Sanicey/notebook/main/2020_denoising_diffusion_probabilistic_models/notebook.ipynb)

- **发表出处**：NeurIPS 2020 (Jonathan Ho, Ajay Jain, Pieter Abbeel)
- **核心创新**：固定线性方差前向扩散与闭式任意步采样、Rao-Blackwellized 低方差变分下界分解、固定各向同性反向方差边界、$\epsilon$-预测重参数化与去噪分数匹配/朗之万动力学理论等价、加权简化目标 $L_{\text{simple}}$、$[-1, 1]$ 离散高斯解码器 $L_0$、时间正弦嵌入 + GroupNorm + $16\times 16$ 自注意力 U-Net 与 EMA、渐进有损解压与广义自回归解码、潜空间粗到细插值（9 项全量拆解 + 7 大自包含可运行 PyTorch 积木块）
- **深度解构文件**：[notebook.ipynb](notebook.ipynb)

> 包含严谨公式推导、**内嵌预渲染高清架构图 (`image/png`)**、**7 个严格单格自包含可运行 Mock 代码单元**（涵盖线性噪声调度与闭式后验、VLB 权重与 $L_{\text{simple}}$ 对比、256 级离散高斯解码器、正弦时间嵌入与 GroupNorm 残差块、多尺度 $16\times 16$ 自注意力 U-Net 与 EMA、朗之万式反向采样与渐进 $\hat{x}_0$ 重建、随机潜空间粗到细插值，全部通过 `nbformat 4.5` 与 AST 顺序流 + `symtable` 校验）、精准锚定原论文 Table 1–4 与 Appendix A–D 的消融证据链，以及经文献核验的 **2021–2026** 顶会演进路线（DDIM、Score-SDE、Improved DDPM、ADM & CFG、LDM、EDM、Flow Matching、Consistency Models、DiT、SD3 MMDiT、sCM/TrigFlow ICLR'25、REPA ICLR'25 Oral、Shortcut Models ICLR'25 Oral、MeanFlow NeurIPS'25 Oral、AlphaFlow ICLR'26 & Decoupled/Improved MeanFlow CVPR'26）。请点击上方徽章或直接查看 [notebook.ipynb](notebook.ipynb)。

---

## 模型整体拓扑图（GitHub 原生 Mermaid 视图）

```mermaid
graph TD
    subgraph Forward["前向扩散过程 Forward Diffusion (Fixed Linear beta_t)"]
        X0["Clean Image x_0 in [-1, 1]"] -->|"Step-by-step q(x_t | x_{t-1})"| XT1["Latent x_{t-1}"]
        XT1 -->|"q(x_t | x_{t-1})"| XT["Noisy Latent x_t"]
        XT -->|"T - t steps"| XPrior["Isotropic Prior x_T ~ N(0, I)"]
        X0 -.->|"Closed-Form Arbitrary-t: sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps"| XT
    end

    subgraph Backbone["共享权重时间条件 U-Net Backbone eps_theta(x_t, t)"]
        TimeIn["Timestep t -> Sinusoidal PE + 2-Layer MLP"] -.->|"+ t_emb into every ResBlock"| Enc32
        TimeIn -.->|"+ t_emb"| Enc16
        TimeIn -.->|"+ t_emb"| Bottleneck
        XT --> Enc32["Encoder 32x32 (2x ResBlock + GroupNorm)"]
        Enc32 --> Enc16["Encoder 16x16 (ResBlock + Self-Attention)"]
        Enc16 --> Bottleneck["Bottleneck 8x8 -> 4x4 (Res + Attn + Res)"]
        Bottleneck --> Dec16["Decoder 16x16 (Concat Skip + Self-Attention)"]
        Enc16 -.->|"Skip 16x16"| Dec16
        Dec16 --> Dec32["Decoder 32x32 (Concat Skip + ConvOut)"]
        Enc32 -.->|"Skip 32x32"| Dec32
        Dec32 --> EpsOut["Predicted Noise eps_theta(x_t, t)"]
    end

    subgraph DualFlow["训练目标 vs 反向采样与解码 Dual Flow"]
        EpsOut ==>|"Algorithm 1 Training"| LossSimple["L_simple = ||eps - eps_theta(x_t, t)||^2 (Weighted DSM)"]
        EpsOut ==>|"Algorithm 2 Sampling (EMA=0.9999)"| RevStep["x_{t-1} = (x_t - coef * eps_theta) / sqrt(alpha_t) + sigma_t z"]
        RevStep -->|"t = T -> 1 (z=0 at t=1)"| DecoderL0["Discretized Gaussian Decoder p_theta(x_0 | x_1) & L_0"]
    end
```

## 后续顶会顶刊演进路线图（2020 → 2026）

```mermaid
flowchart LR
    DDPM["2020: DDPM (Ho et al., NeurIPS'20)<br/>eps-Prediction + L_simple + U-Net"] --> Y2021A["2021: DDIM & Score-SDE<br/>(ICLR'21)<br/>非马尔可夫加速 & 连续 SDE/ODE 统一"]
    DDPM --> Y2021B["2021: Improved DDPM & ADM / CFG<br/>(ICML'21 / NeurIPS'21)<br/>余弦调度/学习方差 & 无分类器引导"]
    Y2021B --> Y2022A["2022: Latent Diffusion (LDM)<br/>(Rombach et al., CVPR'22)<br/>VAE 潜空间扩散 + 跨模态交叉注意力"]
    Y2021A --> Y2022B["2022: EDM<br/>(Karras et al., NeurIPS'22)<br/>解耦设计空间与二阶 Heun ODE"]
    Y2022A --> Y2023A["2023: DiT & Flow Matching<br/>(ICCV'23 / ICLR'23)<br/>adaLN-Zero 纯 ViT & 直线速度场"]
    Y2022B --> Y2023B["2023: Consistency Models<br/>(Song et al., ICML'23)<br/>ODE 轨迹原点一致映射迈向少步生成"]
    Y2023A --> Y2024["2024: Stable Diffusion 3 (MMDiT)<br/>(Esser et al., ICML'24 Best Paper)<br/>Rectified Flow + 双向图文多模态 DiT"]
    Y2023B --> Y2025A["2025: sCM & TrigFlow<br/>(Lu & Song, ICLR'25)<br/>1.5B 连续一致性 2 步 50x 加速"]
    Y2024 --> Y2025B["2025: REPA<br/>(Yu et al., ICLR'25 Oral)<br/>DINOv2 表征对齐提速训练 17.5x"]
    Y2024 --> Y2025C["2025: Shortcut Models & MeanFlow<br/>(ICLR'25 Oral / NeurIPS'25 Oral)<br/>步长条件化 & JVP 平均速度从头 1-NFE"]
    Y2025C --> Y2026["2026: AlphaFlow & iMF<br/>(ICLR'26 / CVPR'26)<br/>退火消除梯度冲突 & 解耦改进 MeanFlow"]
```
