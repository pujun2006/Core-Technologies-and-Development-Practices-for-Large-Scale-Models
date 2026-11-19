import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 定义模型名称和数据
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# 2. 准备示例数据集
data = {
    "text": [
        "Hugging Face makes NLP easier.",
        "Transformers are powerful tools.",
        "Custom training loops allow flexibility.",
    ],
    "label": [1, 1, 0],
}
dataset = Dataset.from_dict(data)


# 3. 数据预处理函数（干净的标准写法，不在 map 内部手动转 Tensor）
def preprocess_data(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=32
    )
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs


# 应用预处理并转为 PyTorch 张量格式
dataset = dataset.map(preprocess_data, batched=True)
dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# 4. 使用 DataLoader 构建 Data Batch（非常重要！）
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
eval_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# 5. 设备与优化器配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)


# 6. 自定义训练循环
def train_model(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # 将 Batch 中的所有数据搬运到对应设备 (CPU/GPU)
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播 (BERT 内部会自动用 inputs['labels'] 计算 Loss)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch {epoch+1}, Batch {i+1} - Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} 平均损失: {avg_loss:.4f} ---")


# 7. 自定义评估函数
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


# 8. 开始训练与评估
train_model(model, train_dataloader, optimizer, epochs=3)
evaluate_model(model, eval_dataloader)