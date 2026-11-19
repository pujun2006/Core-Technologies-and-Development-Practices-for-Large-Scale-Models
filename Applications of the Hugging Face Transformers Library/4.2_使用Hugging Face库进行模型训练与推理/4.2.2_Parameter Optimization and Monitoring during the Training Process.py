import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 1. 加载模型与 Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# 2. 示例数据
texts = [
    "I love Hugging Face!",
    "Transformers library is amazing.",
    "Parameter optimization is crucial.",
]
labels = [1, 1, 0]


# 3. 规范的数据预处理函数
def preprocess_data(texts, labels):
    # 提取 List 格式的 input_ids 与 attention_mask，避免 Tensor 包装嵌套
    inputs = tokenizer(
        texts, padding="max_length", truncation=True, max_length=32
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }
    )
    return dataset


# 预处理后的数据集
dataset = preprocess_data(texts, labels)

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # ✅ 使用更新后的 eval_strategy 替代 evaluation_strategy
    save_strategy="epoch",  # 保存策略与评估策略保持一致
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,  # 权重衰减
    logging_steps=10,  # 每 10 步记录一次日志
    report_to="none",  # 禁用默认的 wandb 提交提示，保持终端干净
)


# 5. 定义评估指标计算函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}


# 6. 初始化 Trainer 并开始训练
trainer = Trainer(
    model=model,  # 传入模型
    args=training_args,  # 传入训练参数
    train_dataset=dataset,  # 训练集
    eval_dataset=dataset,  # 验证集
    compute_metrics=compute_metrics,  # 评估指标
)

# 开始训练
trainer.train()