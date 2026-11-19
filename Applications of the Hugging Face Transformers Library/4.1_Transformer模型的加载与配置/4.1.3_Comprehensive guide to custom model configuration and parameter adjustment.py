import torch
from torch.optim import AdamW  # ✅ 使用 PyTorch 原生的 AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"

# 1. 加载 Tokenizer 与模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 在加载模型时直接传入 Dropout 参数，确保更改立即生效
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 用于二分类任务
    hidden_dropout_prob=0.3,  # 隐藏层 Dropout
    attention_probs_dropout_prob=0.3,  # 注意力 Dropout
)
print("模型和 Tokenizer 加载成功！")
print("生效后的 Dropout 概率:", model.config.hidden_dropout_prob)

# 2. 准备示例数据
sentences = [
    "Hugging Face is transforming NLP!",
    "The transformers library is very popular.",
]
labels = [1, 0]  # 假设 1 表示正面，0 表示负面

# 将文本转化为模型输入格式
inputs = tokenizer(
    sentences, padding=True, truncation=True, return_tensors="pt"
)

# 将 labels 转化为 Tensor 并加入到 inputs 字典中
inputs["labels"] = torch.tensor(labels)

# 3. 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
print("自定义优化器配置完成")

# 4. 模拟训练步骤
model.train()  # 切换到训练模式
for epoch in range(3):  # 进行 3 个 epoch
    optimizer.zero_grad()  # 梯度清零

    # 获取模型输出（传入 labels 后，model 会自动计算 CrossEntropyLoss）
    outputs = model(**inputs)

    loss = outputs.loss  # ✅ 直接提取内置计算出的损失
    logits = outputs.logits

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    # 反向传播和优化
    loss.backward()
    optimizer.step()

# 5. 保存微调后的模型和 Tokenizer
model.save_pretrained("./finetuned_bert")
tokenizer.save_pretrained("./finetuned_bert")
print("\n微调后的模型已保存至 ./finetuned_bert")

# 6. 从保存的目录加载模型并进行推理
finetuned_model = AutoModelForSequenceClassification.from_pretrained(
    "./finetuned_bert"
)
finetuned_tokenizer = AutoTokenizer.from_pretrained("./finetuned_bert")
print("微调后的模型加载成功！")

# 推理环节
finetuned_model.eval()  # ✅ 推理前切回评估模式（关闭 Dropout）
test_sentence = "This library simplifies NLP tasks."
test_inputs = finetuned_tokenizer(test_sentence, return_tensors="pt")

with torch.no_grad():  # 推理阶段无需计算梯度，节省内存
    outputs = finetuned_model(**test_inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

print(f"测试文本: '{test_sentence}'")
print(f"预测类别: {predicted_label} (各类别概率: {probabilities.tolist()[0]})")