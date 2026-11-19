import numpy as np
import torch
from torch.optim import AdamW  # ✅ 弃用 transformers.AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# 1. 定义模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 2. 示例数据：旧数据与新数据
old_texts = ["旧数据样本1", "旧数据样本2", "旧数据样本3"]
new_texts = ["新数据样本1", "新数据样本2"]
old_labels = [0, 1, 0]
new_labels = [1, 2]

texts = old_texts + new_texts
labels = old_labels + new_labels


# 3. 规范的 CustomTextDataset（仅截断，不硬编码 Padding）
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": str(self.texts[idx]),
            "labels": self.labels[idx],
        }


# 4. 数据整理函数（Collator）
def collate_fn(batch):
    batch_texts = [item["text"] for item in batch]
    batch_labels = [item["labels"] for item in batch]

    # 动态 Padding
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(batch_labels, dtype=torch.long)
    return inputs


dataset = CustomTextDataset(texts, labels)

# 5. 权重设置（新数据权重 1.0，旧数据权重 0.5）
weights = [0.5 if i < len(old_texts) else 1.0 for i in range(len(labels))]

# 实例化 WeightedRandomSampler
# replacement=True 意味着采样是放回抽样，权重高的样本在单个 Epoch 中可能被多次抽到
sampler = WeightedRandomSampler(
    weights=weights, num_samples=len(weights), replacement=True
)

# 6. 创建 DataLoader（注意：有了 sampler，不能设置 shuffle=True）
dataloader = DataLoader(
    dataset, sampler=sampler, batch_size=2, collate_fn=collate_fn
)

# 7. 优化器与模型迁移
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 8. 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels
        )
        loss = outputs.loss
        loss.backward()

        # 梯度裁剪防爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"第 {epoch+1} 轮平均损失: {avg_loss:.4f}")