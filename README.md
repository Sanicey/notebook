# 📚 AI Paper-to-Notebook Knowledge Base

> 本仓库汇集前沿顶级学术成果（NeurIPS / ICML / ICLR / CVPR / IEEE TPAMI），提供**全量创新点提炼**、**数学公式符号严谨定义**、**自包含可运行 PyTorch 积木块 (Runnable Mock)**、**消融实验量化证据链**及**前沿改进方案**。支持 Google Colab 与 Kaggle 双云端一键即点即跑。

---

<!-- STATS_START -->
- 📚 **已收录论文**：`1` 篇
- 🏛️ **发表出处分布**：NeurIPS 2017 (1)
- 🕒 **最近同步**：2026-11-21
<!-- STATS_END -->

---

## 🧭 论文深度解构与可运行 Notebook 导航总表

| 序号 | 论文名称 (Title) | 发表会议/期刊 | 全量核心创新点速览 | 可执行文件 | 云端一键直达 (Colab / Kaggle) |
| :---: | :--- | :---: | :--- | :---: | :---: |
| #001 | **Attention Is All You Need** | NeurIPS 2017 | 纯注意力无循环无卷积架构、缩放点积注意力、多头注意力、正弦位置编码、Post-LN、因果掩码、交叉注意力、Noam 调度与正则化组合等 9 项全量创新 | [📖 Notebook](2017_attention_is_all_you_need/notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanicey/notebook/blob/main/2017_attention_is_all_you_need/notebook.ipynb) &nbsp; [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Sanicey/notebook/main/2017_attention_is_all_you_need/notebook.ipynb) |
<!-- PAPER_TABLE_END -->

---

## 🛠️ 单篇 Notebook 工程质量硬性标准
1. **全量创新点穷尽**：系统通读全文，不遗漏任何工程机制与理论贡献；
2. **纯净自包含代码**：每个子模块自带输入 Mock 张量与 `assert` 维度检查，无隐式外部依赖；
3. **消融实验证据链**：严格锚定原论文消融表格具体序号与量化指标变化；
4. **严谨文献溯源**：如实引用顶刊/顶会改进成果（含真实作者、会议/期刊与年份），严禁虚构出处；
5. **自动化质检门禁**：编译产物须通过 `nbformat 4.5`、`symtable` 独立性检查与 GitHub Actions CI 冒烟运行。
