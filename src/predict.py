import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import os


class GarbageClassifier:
    def __init__(self, model_path="../model/roberta_garbage_model", data_path="../data/garbage_sorting.csv"):
        # 1. 【核心】构建精准规则库
        # 将训练集里所有数据加载到内存字典中
        print("正在构建规则库...")
        self.rules = {}
        df = pd.read_csv(data_path, encoding='utf-8') if os.path.exists(data_path) else pd.DataFrame()
        # 建立映射：名字 -> 类别名
        type_map = {1: "可回收物", 2: "其他垃圾", 3: "厨余垃圾", 4: "有害垃圾"}
        for _, row in df.iterrows():
            name = str(row['garbage_name']).strip()
            t_id = int(row['type'])
            self.rules[name] = type_map.get(t_id, "未知")

        # 2. 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        text = text.strip()

        # === 第一道防线：规则库 (解决“指甲油瓶”等硬规则) ===
        # 如果这个词在我们的 Excel/CSV 里出现过，直接返回标准答案，别让 AI 猜
        if text in self.rules:
            return self.rules[text], "精准规则 (100%置信度)"

        # === 第二道防线：AI 模型 (解决未知新词) ===
        # 只有规则库里没有的词，才让 AI 跑
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
        label = self.model.config.id2label[pred_id]

        return label, f"AI推断 ({confidence:.2%})"

if __name__ == "__main__":
    clf = GarbageClassifier()
    print(clf.predict("指甲油瓶"))
    print(clf.predict("陶瓷水壶"))