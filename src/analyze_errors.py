import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 配置
MODEL_PATH = "../model/roberta_garbage_model"
DATA_PATH = "../data/garbage_sorting.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDataset(Dataset):
    def __init__(self, encodings, labels, texts):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item, self.texts[idx]

    def __len__(self):
        return len(self.labels)


def analyze():
    print(f"正在加载模型... ({DEVICE})")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print("错误：找不到模型文件，请检查路径 '../model/roberta_garbage_model'")
        return

    model.to(DEVICE)
    model.eval()

    # 加载数据 (必须和训练时完全一致的切分逻辑)
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
    except:
        df = pd.read_csv(DATA_PATH, encoding='gbk')

    df['garbage_name'] = df['garbage_name'].astype(str)
    df['label'] = df['type'].astype(int) - 1

    # 切分验证集
    _, val_texts, _, val_labels = train_test_split(
        df['garbage_name'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # 编码
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)
    dataset = SimpleDataset(val_encodings, val_labels, val_texts)
    loader = DataLoader(dataset, batch_size=32)

    id2label = {0: "可回收", 1: "其他(干)", 2: "厨余(湿)", 3: "有害"}

    print("开始分析验证集错误样本...")
    errors = []

    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = {k: v.to(DEVICE) for k, v in batch[0].items() if k != 'labels'}
            labels = batch[0]['labels'].to(DEVICE)
            texts = batch[1]

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    errors.append({
                        "垃圾名称": texts[i],
                        "真实类别": id2label[labels[i].item()],
                        "预测类别": id2label[preds[i].item()]
                    })

    # 打印前 20 个错误
    print("\n" + "=" * 60)
    print(f"验证集总数: {len(val_labels)} | 错误数: {len(errors)} | 准确率: {(1 - len(errors) / len(val_labels)):.4f}")
    print("=" * 60)
    print(f"{'垃圾名称':<20} | {'真实类别':<10} | {'预测类别':<10}")
    print("-" * 60)

    for err in errors[:30]:  # 展示前30个
        print(f"{err['垃圾名称']:<20} | {err['真实类别']:<10} | {err['预测类别']:<10}")

    # 保存所有错误到 CSV 方便查看
    if errors:
        pd.DataFrame(errors).to_csv("bad_cases.csv", index=False, encoding='utf-8-sig')
        print(f"\n[提示] 所有错误样本已保存至 'src/bad_cases.csv'，请打开查看具体原因！")


if __name__ == "__main__":
    analyze()