# Image Classification Pipeline (Metwant Exam)

这是一个基于 CLIP 模型的图像分类工程实现，旨在处理存储在 TAR 归档中的大规模图像数据，并利用 Parquet 索引进行高效读取。

## 🛠️ 项目环境与安装

本项目使用 Python 3.12 开发，推荐使用 `uv` 或 `pip` 进行依赖管理。

### 前置要求
- Python >= 3.10
- 内存 >= 8GB (推荐)

### 安装依赖
```bash
# 如果使用 uv (推荐)
uv sync

# 或者使用 pip
pip install -r requirements.txt
```

## 🚀 快速开始

### 数据准备
请确保数据文件放置在 `data/` 目录下：
- `data/zip1.tar`: 原始图像归档文件
- `data/index.parquet`: 包含 offset 和 size 的索引文件

### 运行推理
```bash
# 默认运行
uv run src/run.py

# 指定参数运行
uv run src/run.py --tar_path ./data/zip1.tar --batch_size 64
```

## 💡 核心思考链路与工程决策 (Design & Thinking Process)

在完成本次 Exam 的过程中，我主要关注以下三个维度的工程实现与优化：

### 1. 数据 I/O 的高效性 (Data Efficiency)
*   **问题**：通常图像数据集需要解压后读取，但这会占用大量 inode 和磁盘空间，且小文件读取速度慢。
*   **解决方案**：实现了 `TarImageDataset` 类。利用 Parquet 提供的 `offset` (字节偏移量) 和 `size`，直接在 TAR 文件流中进行 `seek()` 和 `read()`。
*   **优势**：实现了**零解压 (Zero-extraction)** 读取，极大降低了磁盘 I/O 开销，支持随机访问。
*   **容错处理**：针对文件名匹配问题（如 `zip1` vs `zip1.tar`），在 Dataset 中增加了后缀兼容逻辑，增强了代码的鲁棒性。

### 2. 模型推理与架构 (Model & Inference)
*   **模型选择**：选用 `openai/clip-vit-base-patch32`。
    *   *理由*：CLIP 具备强大的 Zero-shot 能力，无需重新训练即可通过文本 Prompt 对图像进行分类，非常适合题目中这种开放类别的场景。
*   **批处理 (Batch Processing)**：
    *   使用 `DataLoader` 进行 Batch 读取，而非单张处理，充分利用 CPU/GPU 的并行计算能力。
    *   实现了 `collate_fn` 逻辑（通过 Dataset 返回字典），过滤掉损坏的图像数据，防止因单张坏图导致整个 Batch 崩溃。

### 3. 跨平台兼容性与性能优化 (Compatibility & Optimization)
*   **Mac MPS 加速适配**：
    *   在代码中检测 `torch.backends.mps.is_available()`。虽然 MPS (Metal Performance Shaders) 能加速推理，但其对 `pin_memory` 支持尚不完善。
    *   **决策**：在检测到 MPS 环境时自动关闭 `pin_memory` 并禁用 Tokenizers 并行，消除了 `fork` 警告和潜在的死锁风险，确保在开发机（MacBook）上的流畅运行。

## 📂 项目结构

```text
.
├── data/               # 数据存放目录 (已在 .gitignore 中忽略)
├── src/
│   ├── dataset.py      # TarImageDataset 实现 (核心 I/O 逻辑)
│   ├── predictor.py    # CLIP 模型封装
│   └── run.py          # 程序入口与 CLI
├── README.md           # 项目文档
├── pyproject.toml      # 依赖配置
└── requirements.txt    # 依赖列表
```

## 📊 结果示例

运行完成后，结果将保存在 `results.csv` 中，包含以下字段：
- `key`: 图像唯一标识
- `label`: 预测类别
- `score`: 置信度分数

---
*Author: Johnny Lingsong Wang*

