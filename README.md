# 🗑️ Intelligent Garbage Classification System based on RoBERTa
# 基于 RoBERTa 的智能垃圾分类系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Transformer](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)

[中文文档](#-中文文档) | [English Documentation](#-english-documentation)

---

## 📖 中文文档

### 项目简介
这是一个基于深度学习（NLP）的智能垃圾分类系统。项目使用 **RoBERTa-wwm-ext** 预训练模型在垃圾分类数据集上进行微调（Fine-tuning）。

**核心亮点：**
*   **混合预测模式 (Hybrid Mode)**：采用“规则库 + AI模型”的双重策略。对于数据库中已有的垃圾，直接查表返回（100% 准确）；对于未知的新词，使用 AI 进行语义推理。
*   **高性能模型**：使用哈工大讯飞的 `chinese-roberta-wwm-ext` 模型，在中文语境下表现优异。
*   **生产级接口**：基于 **FastAPI** 封装了高性能的 HTTP 接口，支持 GET 请求调用。
*   **数据清洗工具**：内置数据冲突检测脚本，防止“脏数据”影响模型训练。

### 📂 目录结构
```text
garbage_classification/
├── data/
│   └── garbage_sorting.csv    # 原始数据集 (id, garbage_name, type)
├── src/
│   ├── train.py               # 模型训练脚本 (Fine-tuning)
│   ├── predict.py             # 命令行预测脚本 (包含规则+模型)
│   ├── api_server.py          # FastAPI 接口服务
│   ├── check_data.py          # 数据质量与冲突检查脚本
│   └── analyze_errors.py      # 错误案例分析工具
├── model/                     # 训练好的模型保存路径 (Git忽略)
├── utils/                     # 通用工具包
└── requirements.txt           # 依赖列表
```

### 🛠️ 环境安装

1.  **克隆项目**
    ```bash
    git clone https://github.com/your-username/garbage_classification.git
    cd garbage_classification
    ```

2.  **安装依赖**
    建议使用 Conda 或 venv 虚拟环境。需安装 PyTorch 和 Transformers 等库。
    ```bash
    pip install torch transformers pandas scikit-learn fastapi uvicorn accelerate
    ```

### 🚀 快速开始

#### 1. 数据检查 (必做)
训练前必须运行此脚本，确保 CSV 中不存在“同名不同类”的冲突数据。
```bash
python src/check_data.py
```

#### 2. 模型训练
运行训练脚本。程序会自动下载预训练模型，并将微调后的模型保存至 `../model/roberta_garbage_model`。
```bash
python src/train.py
```
*提示：建议使用 NVIDIA GPU 进行训练，速度会快很多。*

#### 3. 命令行测试
在终端中交互式测试垃圾分类效果。
```bash
python src/predict.py
```

#### 4. 启动 API 服务
启动后端服务，供前端或小程序调用。
```bash
python src/api_server.py
```
*   服务地址: `http://0.0.0.0:9000`
*   接口文档: `http://127.0.0.1:9000/docs`

### 🔌 API 接口说明

**接口地址:** `GET /predict`

**参数:**
*   `text` (string): 需要查询的垃圾名称

**调用示例:**
浏览器访问或使用代码请求：
`http://127.0.0.1:9000/predict?text=香蕉皮`

**返回示例:**
```json
{
  "name": "香蕉皮",
  "type": 3,
  "confidence": "100.00%",
  "source": "rule_match",
  "desc": "厨余垃圾"
}
```
*   `source` 为 `rule_match` 表示命中规则库（精准）；为 `ai_predict` 表示由 AI 推理。

### 🧩 垃圾分类标准
本项目数据遵循以下分类标准（可根据数据集调整）：
*   **Type 1**: 可回收物 (Recyclable)
*   **Type 2**: 其他垃圾 / 干垃圾 (Residual / Other)
*   **Type 3**: 厨余垃圾 / 湿垃圾 (Kitchen / Food Waste)
*   **Type 4**: 有害垃圾 (Hazardous)

### ⚠️ 注意事项
*   模型权重文件较大（约 400MB），因此 `model/` 文件夹已被 `.gitignore` 忽略，请在本地运行 `train.py` 生成模型。
*   数据决定了分类标准，请确保 `data/garbage_sorting.csv` 中的数据符合您当地的分类法规。

---

## 📖 English Documentation

### Introduction
This project is an intelligent garbage classification system powered by Deep Learning (NLP). It utilizes the **RoBERTa-wwm-ext** pre-trained model to fine-tune on garbage classification datasets.

**Key Features:**
*   **Hybrid Prediction Logic:** Combines an **Exact Match Rule Base** (100% accuracy for known data) with an **AI Model** (high generalization for unknown data).
*   **High Performance:** Uses `hfl/chinese-roberta-wwm-ext` for state-of-the-art Chinese text classification.
*   **Production Ready API:** Provides a high-performance REST API using **FastAPI**.
*   **Data Integrity:** Includes scripts for data validation and conflict detection.

### 📂 Project Structure
```text
garbage_classification/
├── data/
│   └── garbage_sorting.csv    # Dataset (id, garbage_name, type)
├── src/
│   ├── train.py               # Model training script (Fine-tuning)
│   ├── predict.py             # CLI prediction script (Hybrid mode)
│   ├── api_server.py          # FastAPI server
│   ├── check_data.py          # Data quality & conflict checker
│   └── analyze_errors.py      # Error analysis on validation set
├── model/                     # Directory for saving trained models (Excluded from git)
├── utils/                     # Utility functions
└── requirements.txt           # Dependencies
```

### 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/garbage_classification.git
    cd garbage_classification
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install torch transformers pandas scikit-learn fastapi uvicorn accelerate
    ```

### 🚀 Usage Guide

#### 1. Data Check (Crucial)
Before training, ensure there are no conflicting labels in your dataset (e.g., same name with different types).
```bash
python src/check_data.py
```

#### 2. Model Training
Fine-tune the RoBERTa model. This will automatically download the pre-trained weights and save the best model to `./model/roberta_garbage_model`.
```bash
python src/train.py
```
*Note: A GPU is highly recommended for training.*

#### 3. CLI Prediction
Test the model in the command line. It uses the Rule Base first, then the AI model.
```bash
python src/predict.py
```

#### 4. Start API Server
Launch the FastAPI server for external access.
```bash
python src/api_server.py
```
*   Server runs at: `http://0.0.0.0:9000`
*   API Docs: `http://127.0.0.1:9000/docs`

### 🔌 API Reference

**Endpoint:** `GET /predict`

**Parameters:**
*   `text` (string): The name of the garbage item.

**Example Request:**
```
http://127.0.0.1:9000/predict?text=BananaPeel
```

**Example Response:**
```json
{
  "name": "BananaPeel",
  "type": 3,
  "confidence": "100.00%",
  "source": "rule_match",
  "desc": "Kitchen Waste"
}
```

### 🧩 Garbage Types
*   **1**: Recyclable (可回收物)
*   **2**: Residual/Other (其他垃圾/干垃圾)
*   **3**: Kitchen/Food Waste (厨余垃圾/湿垃圾)
*   **4**: Hazardous (有害垃圾)

