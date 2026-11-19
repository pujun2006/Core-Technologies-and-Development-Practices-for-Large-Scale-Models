import torch
from torch.optim import AdamW  # 推荐使用 PyTorch 原生 AdamW，避免 DeprecationWarning
from transformers import AutoModelForSequenceClassification

# 1. 定义模型名称并加载预训练分类模型
model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 假设输入数据与标签
input_ids = torch.tensor([[101, 1045, 2572, 1037, 2339, 102]])
labels = torch.tensor([1])

print("=== 1. 执行全参数微调 ===")
# 所有参数可训练
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
optimizer.zero_grad()  # 清空梯度
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print("全参数微调完成：所有层参数已计算梯度并更新。\n")


print("=== 2. 切换为部分参数冻结微调 ===")
# 冻结 Embedding 层与前 6 层 Encoder
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

# 检查可训练参数状态
for name, param in model.named_parameters():
    status = "✅ 可训练" if param.requires_grad else "❄️ 已冻结"
    print(f"{name}: {status}")

# 仅将 requires_grad=True 的参数传给优化器
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
)

# 执行第二次训练迭代
optimizer.zero_grad()  # 训练前清空上一轮梯度
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print("\n部分参数微调完成：仅高层 Encoder 与分类头参数进行了更新。")