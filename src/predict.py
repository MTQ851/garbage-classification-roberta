import torch
import pandas as pd
import os
import jieba
import jieba.posseg as pseg  # 引入词性标注
from transformers import BertTokenizer, BertForSequenceClassification


class GarbageClassifier:
    def __init__(self, model_path="../model/roberta_garbage_model", data_path="../data/garbage_sorting.csv"):
        # =================================================
        # 1. 构建精准规则库 & 同步 Jieba 词库 (核心修改点)
        # =================================================
        print("正在构建规则库并初始化分词器...")
        self.rules = {}

        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except:
                df = pd.read_csv(data_path, encoding='gbk')

            # 建立映射：数值 -> 类别名
            type_map = {1: "可回收物", 2: "其他垃圾", 3: "厨余垃圾", 4: "有害垃圾"}

            for _, row in df.iterrows():
                name = str(row['garbage_name']).strip()
                t_id = int(row['type'])

                # A. 存入规则字典
                self.rules[name] = type_map.get(t_id, "未知")

                # B. 【关键】告诉 jieba 这是一个专有名词，不要切开！
                # freq=100000 强行提高词频，tag='n' 强行标记为名词
                jieba.add_word(name, freq=100000, tag='n')

            print(f"规则库加载完成，共 {len(self.rules)} 条数据。")
        else:
            print("【警告】未找到数据文件，规则库无法使用！")

        # 2. 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载 AI 模型... (设备: {self.device})")

        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"【错误】模型加载失败: {e}")
            # 如果只是想测试规则库，这里可以 pass，但 AI 预测会报错

    def extract_core_noun(self, text):
        """
        从句子中提取核心名词
        例如: "拆开的快递盒" -> 提取出 "快递盒"
        """
        try:
            words = pseg.cut(text)
            nouns = []
            for word, flag in words:
                # 提取名词(n), 名动词(vn)
                if flag.startswith('n') or flag.startswith('vn'):
                    nouns.append(word)

            # 如果提取到了名词，返回最长的那个（通常是主体）
            if nouns:
                return max(nouns, key=len)
        except:
            pass
        return None

    def predict(self, text):
        raw_text = text.strip()
        if not raw_text:
            return "无输入", "0%"

        # ==========================================
        # 第一层：原词直接查表 (最快)
        # ==========================================
        if raw_text in self.rules:
            return self.rules[raw_text], "100% (精准匹配-原词)"

        # ==========================================
        # 第二层：提取名词查表 (解决"拆开的快递盒")
        # ==========================================
        core_noun = self.extract_core_noun(raw_text)

        # 只有当提取出的词和原词不一样，且在规则库里时，才返回
        if core_noun and core_noun != raw_text:
            if core_noun in self.rules:
                return self.rules[core_noun], f"100% (精准匹配-核心词:{core_noun})"

        # ==========================================
        # 第三层：AI 模型预测 (处理生僻词)
        # ==========================================
        # 优先把“核心名词”喂给 AI，干扰更少。如果没提取出名词，就用原句。
        ai_input = core_noun if core_noun else raw_text

        inputs = self.tokenizer(ai_input, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

        # 获取标签名称
        label = self.model.config.id2label[pred_id]

        source_info = f"AI推断 (输入:{ai_input})"
        return label, f"{confidence:.2%} [{source_info}]"


if __name__ == "__main__":
    clf = GarbageClassifier()
    print("-" * 30)

    # 测试案例 1: 规则库里有的
    res, conf = clf.predict("指甲油瓶")
    print(f"输入: 指甲油瓶\n结果: {res} | 置信度: {conf}")
    print("-" * 30)

    # 测试案例 2: 带修饰词的 (之前会出错的)
    res, conf = clf.predict("拆开的快递盒")
    print(f"输入: 拆开的快递盒\n结果: {res} | 置信度: {conf}")
    print("-" * 30)

    # 测试案例 3: AI 推理
    res, conf = clf.predict("外星人的陶瓷水壶")
    print(f"输入: 外星人的陶瓷水壶\n结果: {res} | 置信度: {conf}")
    print("-" * 30)