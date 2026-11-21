import lime
import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载预训练模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 自动选择计算设备 (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# 2. 定义模型预测函数（添加了 Mini-Batch 批处理机制）
def predict_proba(texts, batch_size=32):
    all_probabilities = []

    # 将 LIME 生成的几百条扰动文本分批送入模型，防止 CPU/GPU 内存溢出卡死
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # 移动到相应设备
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            probabilities = F.softmax(outputs.logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    return np.vstack(all_probabilities)


# 3. 初始化 LIME 文本解释器
explainer = LimeTextExplainer(class_names=["negative", "positive"])

# 4. 示例文本
texts = [
    "This movie was fantastic! I loved the storyline and the characters.",
    "I was disappointed with the plot; it was predictable and dull.",
]

# 5. 遍历每个文本示例计算 LIME 解释
for text in texts:
    # num_samples 默认为 5000 次扰动抽样，降低该值（如 500 或 1000）可加速测试
    explanation = explainer.explain_instance(
        text, classifier_fn=predict_proba, num_features=10, num_samples=1000
    )

    # 输出解释结果
    print(f"=" * 50)
    print(f"原文本: '{text}'")
    print("-" * 50)
    print("LIME 特征归因权重:")
    for word, weight in explanation.as_list():
        print(f"  单词: '{word:<15}' | 权重: {weight:.4f}")
    print("\n")