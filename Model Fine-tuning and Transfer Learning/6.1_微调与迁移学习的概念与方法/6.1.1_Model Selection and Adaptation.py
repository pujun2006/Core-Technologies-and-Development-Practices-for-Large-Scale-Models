import torch
from transformers import BertForSequenceClassification

model_name = "bert-base-chinese"

# 1. 直接加载用于 3 分类任务的预训练模型
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)
print(f"已成功加载分类模型: {model_name}\n")

# 2. 正确冻结底层参数 (Embedding + 前 6 层 Transformer Block)
# 注意：BertForSequenceClassification 内部的 Transformer 主体叫做 model.bert
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

# 3. 验证并统计各层参数的可训练状态 (requires_grad)
trainable_params = 0
all_params = 0

print("=== 参数冻结状态检查 ===")
for name, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"✅ 可训练参数 (Trainable): {name}")
    else:
        print(f"❄️ 已冻结参数 (Frozen):    {name}")

print("-" * 60)
print(f"总参数量 (Total Params): {all_params}")
print(
    f"可训练参数量 (Trainable Params): {trainable_params} ({100 * trainable_params / all_params:.2f}%)"
)