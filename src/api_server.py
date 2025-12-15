import os
import torch
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification


# ==========================================
# 1. 核心分类器逻辑 (修复返回值数量)
# ==========================================
class GarbageClassifier:
    def __init__(self):
        # 动态获取路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../model/roberta_garbage_model")
        data_path = os.path.join(base_dir, "../data/garbage_sorting.csv")

        print(f">>> 初始化服务...")

        # 1. 构建规则库 (直接存 int 类型)
        self.rules = {}
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except:
                df = pd.read_csv(data_path, encoding='gbk')

            for _, row in df.iterrows():
                name = str(row['garbage_name']).strip()
                # 存入数值类型
                self.rules[name] = int(row['type'])
            print(f"    规则库加载完成: {len(self.rules)} 条")
        else:
            print("    [警告] 数据文件未找到，仅使用模型模式。")

        # 2. 加载 AI 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    加载推理设备: {self.device}")

        if not os.path.exists(model_path):
            # 如果没有训练好的模型，为了防止API启动失败，这里可以打印警告，或者抛出异常
            # 这里抛出异常提醒你去训练
            raise FileNotFoundError(f"模型路径不存在: {model_path}，请先运行 train.py")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(">>> 服务初始化完成！")

    def predict(self, text):
        """
        核心修复点：确保无论走规则还是走AI，都返回 3 个值
        Returns: (type_int, confidence_str, source_str)
        """
        text = str(text).strip()
        if not text:
            return None, None, None

        # --- 策略A：规则匹配 ---
        if text in self.rules:
            # 【修复】返回 3 个值
            return self.rules[text], "100.00%", "rule_match"

        # --- 策略B：模型预测 ---
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

        # 将模型预测的 0-3 映射回 1-4
        predicted_type = pred_id + 1
        conf_str = f"{confidence * 100:.2f}%"

        # 【修复】返回 3 个值
        return predicted_type, conf_str, "ai_predict"


# ==========================================
# 2. FastAPI 接口定义 (GET 版本)
# ==========================================

class PredictResponse(BaseModel):
    name: str
    type: int  # 数值类型
    confidence: str
    source: str
    desc: str


app = FastAPI(title="垃圾分类 API", version="1.3")

classifier = None

TYPE_DESC_MAP = {
    1: "可回收物",
    2: "干垃圾/其他垃圾",
    3: "湿垃圾/厨余垃圾",
    4: "有害垃圾"
}


@app.on_event("startup")
def load_model():
    global classifier
    classifier = GarbageClassifier()


@app.get("/predict", response_model=PredictResponse)
async def api_predict(text: str):
    if not classifier:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # 这里接收 3 个返回值，如果 predict 方法只返回 2 个就会报错
    label_id, conf, source = classifier.predict(text)

    if label_id is None:
        raise HTTPException(status_code=400, detail="输入不能为空")

    return PredictResponse(
        name=text,
        type=label_id,
        confidence=conf,
        source=source,
        desc=TYPE_DESC_MAP.get(label_id, "未知")
    )


if __name__ == "__main__":
    print("正在启动 API 服务...")
    # 端口可以根据需要修改，这里用 9000
    uvicorn.run(app, host="0.0.0.0", port=9000)