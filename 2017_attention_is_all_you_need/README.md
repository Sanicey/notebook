# Attention Is All You Need

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanicey/notebook/blob/main/2017_attention_is_all_you_need/notebook.ipynb) &nbsp; [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Sanicey/notebook/main/2017_attention_is_all_you_need/notebook.ipynb)

- **发表出处**：NeurIPS 2017 (Vaswani et al.)
- **核心创新**：纯注意力无循环无卷积架构、缩放点积注意力、多头注意力、正弦连续位置编码、Post-LN、因果掩码、交叉注意力、Noam 调度与正则化组合（9 项全量拆解）
- **深度解构文件**：[notebook.ipynb](notebook.ipynb)

> 包含数学公式推导、**严格单格自包含可运行 Mock 代码**（FP16-safe Mask、通过 `nbformat 4.5` 与 `symtable` 校验）、精准锚定原论文 Table 3 的消融证据链，以及经文献核验的 2020–2025 顶会顶刊演进路线（Pre-LN、FlashAttention、DeepNet、Mamba-2、DeepSeek-V2 MLA、Diff Transformer ICLR'25、NSA ACL'25、Titans NeurIPS'25）。请点击上方徽章或直接查看 [notebook.ipynb](notebook.ipynb)。

---

## 模型整体拓扑图（GitHub 原生 Mermaid 视图）

```mermaid
graph TD
    subgraph Inputs["输入端 Input Processing"]
        Src["Source Tokens"] --> EmbSrc["Input Embedding"]
        PosSrc["Sinusoidal Positional Encoding"] --> AddSrc["Elementwise Add (+)"]
        EmbSrc --> AddSrc
    end

    subgraph Encoder["Encoder Stack (N=6)"]
        AddSrc --> EncMHA["Multi-Head Self-Attention"]
        EncMHA --> EncNorm1["Add & LayerNorm"]
        AddSrc -.-> EncNorm1
        EncNorm1 --> EncFFN["Position-wise FFN"]
        EncFFN --> EncNorm2["Add & LayerNorm"]
        EncNorm1 -.-> EncNorm2
    end

    subgraph Outputs["输出端 Output Processing"]
        Tgt["Target Tokens (Shifted Right)"] --> EmbTgt["Output Embedding"]
        PosTgt["Sinusoidal Positional Encoding"] --> AddTgt["Elementwise Add (+)"]
        EmbTgt --> AddTgt
    end

    subgraph Decoder["Decoder Stack (N=6)"]
        AddTgt --> DecMaskMHA["Masked Multi-Head Attention"]
        DecMaskMHA --> DecNorm1["Add & LayerNorm"]
        AddTgt -.-> DecNorm1

        DecNorm1 --> DecCrossMHA["Multi-Head Cross-Attention"]
        EncNorm2 ==>|"Keys (K) & Values (V)"| DecCrossMHA
        DecCrossMHA --> DecNorm2["Add & LayerNorm"]
        DecNorm1 -.-> DecNorm2

        DecNorm2 --> DecFFN["Position-wise FFN"]
        DecFFN --> DecNorm3["Add & LayerNorm"]
        DecNorm2 -.-> DecNorm3
    end

    subgraph Head["Prediction Head"]
        DecNorm3 --> LinearProj["Linear Projection"]
        LinearProj --> Softmax["Softmax"]
        Softmax --> OutPreds["Next-Token Probabilities"]
    end
```

## 后续顶会顶刊演进时间线

```mermaid
timeline
    title Transformer Architecture Evolutionary Roadmap
    2020      : Pre-LN (Xiong et al., ICML 2020) - 稳定梯度无需强依赖 Warmup
    2022      : FlashAttention (Dao et al., NeurIPS 2022) - IO 感知 SRAM 分块精确注意力
    2024      : RoPE (Su et al., Neurocomputing 2024) - 旋转位置编码
              : DeepNet (Wang et al., IEEE TPAMI 2024) - 残差缩放突破 1000 层
              : Mamba-2 SSD (Dao & Gu, ICML 2024) - 状态空间对偶线性复杂度
              : DeepSeek-V2 MLA (DeepSeek-AI, arXiv 2024) - 低秩 KV 压缩
    2025      : Diff Transformer (Ye et al., ICLR 2025 Oral) - 差分注意力抑制噪声与幻觉
              : Native Sparse Attention (Yuan et al., ACL 2025 Best Paper) - 硬件对齐原生稀疏注意力
              : Titans (Behrouz et al., NeurIPS 2025) - 测试期神经长程记忆突破 2M 上下文
```
