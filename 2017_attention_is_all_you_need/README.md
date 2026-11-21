# Attention Is All You Need

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanicey/notebook/blob/main/2017_attention_is_all_you_need/notebook.ipynb) &nbsp; [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Sanicey/notebook/main/2017_attention_is_all_you_need/notebook.ipynb)

- **发表出处**：NeurIPS 2017 (Vaswani et al.)
- **核心创新**：纯注意力无循环无卷积架构、缩放点积注意力、多头注意力、正弦连续位置编码、Post-LN、因果掩码、交叉注意力、Noam 调度与标签平滑正则化组合（9 项全量拆解 + 6 大自包含可运行 PyTorch 积木块）
- **深度解构文件**：[notebook.ipynb](notebook.ipynb)

> 包含数学公式推导、**内嵌预渲染高清架构图 (`image/png`)**、**6 个严格单格自包含可运行 Mock 代码单元**（涵盖 SDPA、MHA、Sinusoidal PE、Encoder Layer、Masked & Cross-Attention Decoder Layer、Noam Scheduler & Label Smoothing，通过 `nbformat 4.5` 与 AST 顺序感知 + `symtable` 校验）、精准锚定原论文 Table 3 的消融证据链，以及经文献核验的 2020–2025 顶会顶刊演进路线（Pre-LN、FlashAttention、RoPE、DeepNet、Mamba-2、DeepSeek-V2 MLA、Diff Transformer ICLR'25、NSA ACL'25、Titans NeurIPS'25）。请点击上方徽章或直接查看 [notebook.ipynb](notebook.ipynb)。

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

## 后续顶会顶刊演进路线图（2017 → 2025）

```mermaid
flowchart LR
    Base["2017: Transformer<br/>(Vaswani et al., NeurIPS'17)<br/>Post-LN + Full MHA"] --> Y2020["2020: Pre-LN<br/>(Xiong et al., ICML'20)<br/>有界深层梯度免强依赖 Warmup"]
    Y2020 --> Y2022["2022: FlashAttention<br/>(Dao et al., NeurIPS'22)<br/>IO 感知 SRAM 分块精确注意力"]
    Y2022 --> Y2024A["2024: RoPE & DeepNet<br/>(Neurocomputing'24 / TPAMI'24)<br/>旋转相对位置 & 1000层残差缩放"]
    Y2022 --> Y2024B["2024: Mamba-2 & MLA<br/>(ICML'24 / DeepSeek-V2)<br/>SSD 状态对偶 & 低秩 KV 压缩"]
    Y2024A --> Y2025A["2025: Diff Transformer<br/>(Ye et al., ICLR'25 Oral)<br/>差分注意力抵消共模长尾噪声"]
    Y2024B --> Y2025B["2025: Native Sparse Attention<br/>(Yuan et al., ACL'25 Best Paper)<br/>硬件对齐端到端原生稀疏注意力"]
    Y2024B --> Y2025C["2025: Titans<br/>(Behrouz et al., NeurIPS'25)<br/>测试期神经长程记忆突破 2M 上下文"]
```
