import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# 1. 设备检测
device = 0 if torch.cuda.is_available() else -1

# 2. 标签映射
id2label = {0: "负面 (Negative)", 1: "正面 (Positive)"}
label2id = {"负面 (Negative)": 0, "正面 (Positive)": 1}

# 3. 加载 Tokenizer 与 模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

# 4. 原始数据与 Dataset 构建
raw_data = {
    "text": [
        "Hugging Face simplifies NLP tasks.",
        "Datasets provide efficient data management.",
        "Pipelines make model inference easy and flexible.",
    ],
    "label": [1, 1, 0],
}
dataset = Dataset.from_dict(raw_data)


# 5. 数据预处理
def preprocess_data(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=32
    )
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs


dataset = dataset.map(preprocess_data, batched=True)

# 6. 初始化 pipeline
nlp_pipeline = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, device=device
)

# 7. 安全提取纯字符串列表进行批量推理（解决报错的关键）
texts = [str(x) for x in raw_data["text"]]
predictions = nlp_pipeline(texts, batch_size=len(texts))

# 8. 输出结果
print("\n=== 批量推理结果 ===")
for text, pred in zip(texts, predictions):
    print(
        f"文本: {text:<50} | 预测标签: {pred['label']:<18} | 置信度: {pred['score']:.4f}"
    )