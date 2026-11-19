import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW  # ✅ 使用 PyTorch 原生 AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 定义模型名称与加载
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# ----------------------------------------------------
# 2. 冻结 BERT 前 6 层 Transformer Encoder 权重（核心逻辑保留）
# ----------------------------------------------------
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

# 验证冻结结果：查看可训练参数与冻结参数的数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(
    f"模型总参数量: {total_params:,} | 可训练参数量: {trainable_params:,} (已冻结前 6 层)"
)

# 3. 示例数据集
data = {
    "text": [
        "Hugging Face makes NLP easier.",
        "Transfer learning improves model efficiency.",
        "Freezing layers prevents overfitting on small datasets.",
    ],
    "label": [1, 1, 0],
}
dataset = Dataset.from_dict(data)


# 4. 数据预处理（规范写法：不在 map 内硬编码 Tensor 转换）
def preprocess_data(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=32
    )
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs


dataset = dataset.map(preprocess_data, batched=True)
# ✅ 关键步骤：统一设置数据输出格式为 PyTorch Tensor
dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# 5. 构建 DataLoader (解决 Batch 打包与类型识别)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
eval_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# 6. 配置设备与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ 只将 requires_grad=True 的参数传给优化器
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
)


# 7. 自定义训练循环
def train_model(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # 将整批 Batch 张量移动至 GPU/CPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # 反向传播与梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch {epoch+1}, Batch {i+1} - Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} 平均损失: {avg_loss:.4f} ---")


# 8. 自定义评估函数
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n评估精度 (Accuracy): {accuracy * 100:.2f}%")


# 9. 开始迁移微调训练与评估
train_model(model, train_dataloader, optimizer, epochs=3)
evaluate_model(model, eval_dataloader)