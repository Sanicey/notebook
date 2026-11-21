# High-Resolution Image Synthesis with Latent Diffusion Models (LDM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanicey/notebook/blob/main/2022_latent_diffusion_models/notebook.ipynb) &nbsp; [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Sanicey/notebook/main/2022_latent_diffusion_models/notebook.ipynb)

- **发表出处**：CVPR 2022 (Robin Rombach*, Andreas Blattmann*, Dominik Lorenz, Patrick Esser, Björn Ommer)
- **核心创新**：感知压缩与语义压缩两阶段解耦、温和 2D 下采样（$f\in\{4,8\}$）与 KL-reg / 吸收式 VQ-reg 第一阶段自编码器、潜空间首批次标准差缩放与 SNR 校准（`0.18215` 溯源）、潜空间 2D 卷积 U-Net 与 SNR-ELBO 目标 $L_{\text{LDM}}$、领域专家编码器 $\tau_\theta$ 与 Table 16 Spatial Transformer 跨模态交叉注意力、双路条件路由与超越 $256^2$ 的百万像素卷积外推、无注意力高分解码器（$f=4^\dagger$）与半轮 $512^2$ 微调、后验图像/感知引导（Appendix B Eq. 16–17）、潜空间 CFG 与盲超分随机退化管线 LDM-BSR（9 项全量拆解 + 7 大自包含可运行 PyTorch 积木块）
- **深度解构文件**：[notebook.ipynb](notebook.ipynb)

> 包含严谨公式推导与原文笔误勘误、**内嵌预渲染高清架构图 (`image/png`)**、**7 个严格单格自包含可运行 Mock 代码单元**（涵盖 KL-reg / 吸收式 VQ-reg 第一阶段自编码器、首批次标准差缩放与 SNR-ELBO 恒等式校验、领域编码器 $\tau_\theta$ 与 Table 16 Spatial Transformer 交叉注意力、Cross-Attention vs Spatial Concatenation 双路 U-Net 骨干、超越 $256^2$ 卷积外推与无注意力解码器显存对比、潜空间 CFG + DDIM 加速采样闭环、Appendix B 后验图像/感知引导与 LDM-BSR 随机退化管线，全部通过 `nbformat 4.5` 与 AST 顺序流 + `symtable` + FP16 安全掩码校验）、精准锚定原论文 Table 1–18 与 Appendix A–H 的消融/算力证据链，以及经文献核验的 **2022–2026** 顶会演进路线（ControlNet ICCV'23 Best Paper、DiT ICCV'23、LCM、SDXL ICLR'24、SD3 MMDiT ICML'24 Best Paper、DC-AE & SANA ICLR'25、VA-VAE / LightningDiT CVPR'25 Oral、REPA-E ICCV'25、RAE ICLR'26 & JiT CVPR'26）。请点击上方徽章或直接查看 [notebook.ipynb](notebook.ipynb)。

---

## 模型整体拓扑图（GitHub 原生 Mermaid 视图）

```mermaid
graph TD
    subgraph Stage1["Stage 1: 感知压缩自编码器 Perceptual Compression (OpenImages Pretrained)"]
        XIn["Input Image x in R^{H x W x 3}"] --> Enc["Encoder E (Downsample f = 2^m, Optimal f in {4, 8})"]
        Enc --> Reg["Latent Regularization<br/>KL-reg (~1e-6) + Std Rescaling z / sigma<br/>or VQ-reg (Continuous Pre-Quant z, Var ~ 1)"]
        Reg -.->|"Stage 1 Pretrain: L1 + LPIPS + PatchGAN + L_reg"| Dec["Decoder D (Single-Pass Decode)<br/>Absorbs VQ Layer / Attention-Free f=4* Option"]
        Dec --> XRec["Reconstructed / Synthesized Image x_tilde"]
    end

    subgraph Stage2["Stage 2: 语义压缩潜扩散 Latent Diffusion (Frozen Stage 1)"]
        Reg ==>|"Encode z_0 = E(x) in R^{h x w x c}"| Fwd["Fixed Forward Markov Chain<br/>z_t = alpha_t z_0 + sigma_t eps -> z_T ~ N(0, I)"]
        Fwd --> UNet["Time-Conditional 2D U-Net eps_theta(z_t, t, tau_theta(y))<br/>ResBlocks + Table 16 SpatialTransformer Cross-Attention"]
        UNet ==>|"Training (Eq. 3)"| LossLDM["L_LDM = ||eps - eps_theta(z_t, t, tau_theta(y))||^2<br/>Jointly Optimizes U-Net & Domain Encoder tau_theta"]
        UNet ==>|"Inference (DDIM 50-250 steps + CFG)"| Rev["Reverse Denoising z_T -> ... -> z_0<br/>Convolutional Extrapolation up to 1024^2"]
        Rev ==>|"Single-Pass Decode D(z_0)"| Dec
    end

    subgraph Cond["双路条件机制与后验引导 Conditioning & Post-hoc Guidance"]
        SpCond["Spatial-Aligned Inputs<br/>Semantic Map / LR Image / Masked Image + Mask"] -->|"Channel Concatenation"| UNet
        ModalCond["Cross-Modal Inputs y<br/>Text Prompts / Layout (l,b,c) / Class Labels"] --> Tau["Domain-Specific Encoder tau_theta<br/>Unmasked Transformer / Embedding"]
        Tau -->|"K, V Projection into SpatialTransformer"| UNet
        PostHoc["Post-hoc Image & Perceptual Guiding (App. B)<br/>L2 / LPIPS Gradient via T(D(z_0(z_t)))"] -.->|"Test-Time Score Update"| Rev
    end
```

## 后续顶会顶刊演进路线图（2022 → 2026）

```mermaid
flowchart LR
    LDM["2022: Latent Diffusion (LDM)<br/>(Rombach et al., CVPR'22)<br/>感知-语义两阶段解耦 + 跨模态交叉注意力"] --> Y2023A["2023: ControlNet<br/>(Zhang et al., ICCV'23 Best Paper)<br/>零卷积副本注入细粒度空间条件"]
    LDM --> Y2023B["2023: DiT & LCM<br/>(ICCV'23 / arXiv'23)<br/>adaLN-Zero 纯 ViT 缩放 & 潜空间少步一致性"]
    LDM --> Y2024A["2024: SDXL<br/>(Podell et al., ICLR'24)<br/>3x U-Net + 双文本编码器 + Refiner 1024^2"]
    Y2023B --> Y2024B["2024: Stable Diffusion 3 (MMDiT)<br/>(Esser et al., ICML'24 Best Paper)<br/>16ch VAE + Rectified Flow + 双向图文 DiT"]
    Y2024B --> Y2025A["2025: DC-AE & SANA<br/>(Chen / Xie et al., ICLR'25)<br/>残差自编码突破 f=32/64/128 与 4K 线性 DiT"]
    Y2024B --> Y2025B["2025: VA-VAE & LightningDiT<br/>(Yao et al., CVPR'25 Oral)<br/>VF Loss 对齐视觉基础模型破解重建-生成困境 (FID 1.35)"]
    Y2025B --> Y2025C["2025: REPA-E<br/>(Leng et al., ICCV'25)<br/>stopgrad + REPA + EMA BN 解锁 VAE 端到端联合微调 (FID 1.12)"]
    Y2025C --> Y2026A["2026: RAE<br/>(Zheng et al., ICLR'26)<br/>冻结 DINOv2/SigLIP2 语义编码器 + DiT^DH (256/512 FID 1.13)"]
    Y2024B --> Y2026B["2026: JiT<br/>(Li & He, CVPR'26)<br/>流形假设回归纯像素 x-prediction 彻底告别 VAE (FID 1.78)"]
```
