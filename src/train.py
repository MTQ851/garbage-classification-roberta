import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

# ===========================
# 配置区域
# ===========================
DATA_PATH = "../data/garbage_sorting.csv"
OUTPUT_DIR = "../model/roberta_garbage_model"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 使用哈工大 RoBERTa，效果优于 BERT
MAX_LEN = 64
EPOCHS = 4  # 减少轮数以防过拟合
BATCH_SIZE = 16  # 如果显存不够，改小为 8


# ===========================
# 数据集类定义
# ===========================
class GarbageDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def train():
    print(">>> 1. 加载数据...")
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
    except:
        df = pd.read_csv(DATA_PATH, encoding='gbk')

    df['garbage_name'] = df['garbage_name'].astype(str)

    # 标签映射：将 1,2,3,4 转换为 0,1,2,3
    df['label'] = df['type'].astype(int) - 1

    # 划分数据集 (80% 训练, 20% 验证)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['garbage_name'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']  # 保证验证集类别比例均衡
    )

    print(f">>> 2. 下载/加载预训练模型: {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 编码
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = GarbageDataset(train_encodings, train_labels)
    val_dataset = GarbageDataset(val_encodings, val_labels)

    # 标签ID映射配置（保存到模型配置中，方便预测时读取）
    id2label = {0: "可回收物", 1: "其他垃圾", 2: "厨余垃圾", 3: "有害垃圾"}
    label2id = {"可回收物": 0, "其他垃圾": 1, "厨余垃圾": 2, "有害垃圾": 3}

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=id2label,
        label2id=label2id
    )

    print(">>> 3. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        warmup_steps=300,
        weight_decay=0.1,  # 增加权重衰减，防止过拟合
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",  # 每个 epoch 评估一次
        save_strategy="epoch",  # 每个 epoch 保存一次
        load_best_model_at_end=True,  # 训练结束加载最好的模型
        metric_for_best_model="accuracy",
        learning_rate=2e-5,  # 微调的标准学习率
        save_total_limit=2,  # 只保留最近2个模型，节省磁盘
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(">>> 4. 开始训练...")
    trainer.train()

    print(f">>> 5. 评估验证集...")
    eval_result = trainer.evaluate()
    print(f"最终验证集准确率: {eval_result['eval_accuracy']:.4f}")

    print(f">>> 6. 保存模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("完成！")


if __name__ == '__main__':
    train()