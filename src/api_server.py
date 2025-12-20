import os
import torch
import pandas as pd
import uvicorn
import jieba  # 【关键】引入 jieba
import jieba.posseg as pseg  # 【关键】引入词性标注
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertForSequenceClassification


# ==========================================
# 1. 核心分类器逻辑 (已同步 predict.py 的修复)
# ==========================================
class GarbageClassifier:
    def __init__(self):
        # 动态获取路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../model/roberta_garbage_model")
        data_path = os.path.join(base_dir, "../data/garbage_sorting.csv")

        print(f">>> 初始化 API 服务...")

        # 1. 构建规则库 & 同步 Jieba 词库
        self.rules = {}
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except:
                df = pd.read_csv(data_path, encoding='gbk')

            print(f"    正在同步 CSV 数据到分词库...")
            for _, row in df.iterrows():
                name = str(row['garbage_name']).strip()
                t_id = int(row['type'])

                # A. 存入规则库
                self.rules[name] = t_id

                # B. 【API 核心修复】告诉 jieba 这是一个专有名词，不要切开！
                # 只有加上这行，"快递盒"才会被识别为一个整体，而不是"快递"+"盒"
                jieba.add_word(name, freq=100000, tag='n')

            print(f"    规则库加载完成: {len(self.rules)} 条")
        else:
            print("    [警告] 数据文件未找到，仅使用模型模式。")

        # 2. 加载 AI 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    加载推理设备: {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}，请先运行 train.py")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 预热分词器
        print("    预热分词组件...")
        pseg.cut("初始化")
        print(">>> 服务启动成功！")

    def extract_core_noun(self, text):
        """
        核心名词提取逻辑
        """
        try:
            words = pseg.cut(text)
            nouns = []
            for word, flag in words:
                # 提取名词(n)和名动词(vn)
                if flag.startswith('n') or flag.startswith('vn'):
                    nouns.append(word)

            # 返回最长的名词
            if nouns:
                return max(nouns, key=len)
        except:
            pass
        return None

    def predict(self, text):
        """
        Returns: (type_int, confidence_str, source_str)
        """
        raw_text = str(text).strip()
        if not raw_text:
            return None, None, None

        # --- 策略A：原词精准规则匹配 ---
        if raw_text in self.rules:
            return self.rules[raw_text], "100.00%", "rule_match_exact"

        # --- 策略B：提取核心名词后查规则 (解决"拆开了的快递盒") ---
        core_noun = self.extract_core_noun(raw_text)

        # 如果提取出的核心词（如"快递盒"）在规则库里
        if core_noun and core_noun != raw_text:
            if core_noun in self.rules:
                return self.rules[core_noun], "100.00%", f"rule_match_keyword({core_noun})"

        # --- 策略C：AI 模型预测 ---
        # 优先用核心词喂给 AI
        ai_input = core_noun if core_noun else raw_text

        inputs = self.tokenizer(ai_input, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

        predicted_type = pred_id + 1
        conf_str = f"{confidence * 100:.2f}%"

        # 标记来源
        source_desc = f"ai_predict({ai_input})" if ai_input != raw_text else "ai_predict"

        return predicted_type, conf_str, source_desc


# ==========================================
# 2. FastAPI 接口定义
# ==========================================

class PredictResponse(BaseModel):
    name: str
    type: int
    confidence: str
    source: str
    desc: str


app = FastAPI(title="垃圾分类 API", version="1.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None

TYPE_DESC_MAP = {
    1: "可回收物",
    2: "其他垃圾/干垃圾",
    3: "厨余垃圾/湿垃圾",
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
    uvicorn.run(app, host="0.0.0.0", port=9000)