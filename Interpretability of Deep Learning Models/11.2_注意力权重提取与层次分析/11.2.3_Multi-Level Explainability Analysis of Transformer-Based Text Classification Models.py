# 若在 Colab 中首次运行且尚未安装依赖，可先执行以下命令：
# !pip install -q lime shap transformers datasets torch

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
try:
    dataset = load_dataset("stanfordnlp/imdb")
except Exception:
    dataset = load_dataset("imdb")

print("数据集加载成功！结构如下：")
print(dataset)

model_name = "bert-base-uncased"
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)


# 2. 数据预处理
def preprocess(data):
    return tokenizer(
        data["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_datasets = dataset.map(preprocess, batched=True)

# 3. 训练配置
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

sample_text = ["The movie was fantastic and I enjoyed every bit of it!"]


# ======================================================================
# 4. 可解释性方法一：SHAP 归因分析
# ======================================================================
print("\n" + "=" * 50)
print(">>> 开始计算 SHAP 解释...")

pred_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    batch_size=32,
    device=0 if torch.cuda.is_available() else -1,
)

shap_explainer = shap.Explainer(pred_pipeline)
shap_values = shap_explainer(sample_text)

print("output_names:", shap_values.output_names)
print("shap_values.shape:", shap_values.shape)

# 兼容自定义 POSITIVE、默认 LABEL_1 或整数索引
if "POSITIVE" in shap_values.output_names:
    target_key = "POSITIVE"
elif "LABEL_1" in shap_values.output_names:
    target_key = "LABEL_1"
else:
    target_key = shap_values.shape[-1] - 1

print(f"SHAP 计算完成，类别 [{target_key}] 归因结果：")
print(shap_values[0, :, target_key])

# 在 Jupyter / Colab 中如需交互式文本归因可视化，可取消下方注释：
# shap.plots.text(shap_values[0])


# ======================================================================
# 5. 可解释性方法二：LIME 局部特征解释
# ======================================================================
print("\n" + "=" * 50)
print(">>> 开始计算 LIME 解释...")


def predict_proba(texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


class_names = ["negative", "positive"]
lime_explainer = LimeTextExplainer(class_names=class_names)

lime_exp = lime_explainer.explain_instance(
    sample_text[0], classifier_fn=predict_proba, num_features=10, num_samples=1000
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

for layer_idx, layer_attention in enumerate(attentions):
    print(
        f"Layer {layer_idx+1:2d} attention weights shape: {layer_attention.shape}"
    )