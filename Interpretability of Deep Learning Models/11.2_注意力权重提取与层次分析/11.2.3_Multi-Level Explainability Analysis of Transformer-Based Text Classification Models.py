import lime
import numpy as np
import shap
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# 1. 加载数据集与分词器/模型初始化
from datasets import load_dataset

try:
    # 优先尝试使用标准命名空间
    dataset = load_dataset("stanfordnlp/imdb")
except Exception:
    # 备用方案
    dataset = load_dataset("imdb")

print("数据集加载成功！结构如下：")
print(dataset)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)


# 2. 数据预处理
def preprocess(data):
    return tokenizer(
        data["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_datasets = dataset.map(preprocess, batched=True)

# 3. 训练配置（移除已废弃的 logging_dir，添加 logging_steps）
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print(">>> 开始微调模型...")
trainer.train()


# ======================================================================
# 4. 可解释性方法一：SHAP 归因分析
# ======================================================================
print("\n" + "=" * 50)
print(">>> 开始计算 SHAP 解释...")

# 适配 SHAP 的正确做法：创建 text-classification pipeline
pred_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1,
)

# 使用正确的 pipeline 构建 SHAP Explainer
shap_explainer = shap.Explainer(pred_pipeline)

sample_text = ["The movie was fantastic and I enjoyed every bit of it!"]
# SHAP 直接接收文本列表
shap_values = shap_explainer(sample_text)

# 绘制文本归因图 (在 Jupyter Notebook 中可用 shap.plots.text(shap_values[0]))
print("SHAP 计算完成，类别 [POSITIVE] 归因结果：")
print(shap_values[0, :, "POSITIVE"])


# ======================================================================
# 5. 可解释性方法二：LIME 局部局部特征解释
# ======================================================================
print("\n" + "=" * 50)
print(">>> 开始计算 LIME 解释...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# 封装 LIME 专属的预测概率函数 (接收文本列表，输出概率矩阵 [N, num_labels])
def predict_proba(texts):
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = F.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()


class_names = ["negative", "positive"]
lime_explainer = LimeTextExplainer(class_names=class_names)

lime_exp = lime_explainer.explain_instance(
    sample_text[0], classifier_fn=predict_proba, num_features=10
)

print("LIME 单词权重解释排序：")
for word, weight in lime_exp.as_list():
    print(f"  单词: '{word:<12}' | 权重: {weight:.4f}")


# ======================================================================
# 6. 可解释性方法三：Layer-wise Attention 权重提取
# ======================================================================
print("\n" + "=" * 50)
print(">>> 开始提取 Layer-wise Attention 权重...")

sample_encoding = tokenizer(
    sample_text, return_tensors="pt", padding=True, truncation=True
).to(device)

input_ids = sample_encoding["input_ids"]
attention_mask = sample_encoding["attention_mask"]

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    attentions = outputs.attentions

# 修正语法错误，避免控制台刷屏
for layer_idx, layer_attention in enumerate(attentions):
    print(
        f"Layer {layer_idx+1:2d} attention weights shape: {layer_attention.shape}"
    )