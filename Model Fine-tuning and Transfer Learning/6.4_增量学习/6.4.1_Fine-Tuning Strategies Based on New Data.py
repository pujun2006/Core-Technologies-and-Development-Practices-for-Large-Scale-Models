import itertools
import torch
import torch.nn.functional as F
from torch.optim import AdamW  # ✅ 使用 PyTorch 原生 AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 1. 定义模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 2. 数据定义
old_texts = ["旧数据样本1", "旧数据样本2", "旧数据样本3"]
new_texts = ["新数据样本1", "新数据样本2"]
old_labels = [0, 1, 0]
new_labels = [1, 2]


# 3. 自定义 Dataset（轻量化，不硬编码 Padding）
class CustomTextDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": str(self.texts[idx]), "labels": self.labels[idx]}


# 4. Collator 动态 Padding 函数
def collate_fn(batch):
    batch_texts = [item["text"] for item in batch]
    batch_labels = [item["labels"] for item in batch]

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(batch_labels, dtype=torch.long)
    return inputs


# 5. 构建两个 DataLoader
old_dataset = CustomTextDataset(old_texts, old_labels)
new_dataset = CustomTextDataset(new_texts, new_labels)

old_dataloader = DataLoader(
    old_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)
new_dataloader = DataLoader(
    new_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

# 6. 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3
alpha = 0.5  # 新旧 Loss 的平衡权重

# 7. 改进后的训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    # ✅ 核心修复 1：使用 cycle 循环较短的数据集，确保长数据集（旧数据）能够完整遍历完所有 Batch
    # 如果旧数据较多，以 old_dataloader 为主主导循环
    steps_per_epoch = len(old_dataloader)
    new_dataloader_iter = itertools.cycle(new_dataloader)

    for old_batch in old_dataloader:
        new_batch = next(new_dataloader_iter)

        # 提取 Batch 并送入 GPU
        old_ids, old_mask, old_lbls = (
            old_batch["input_ids"].to(device),
            old_batch["attention_mask"].to(device),
            old_batch["labels"].to(device),
        )
        new_ids, new_mask, new_lbls = (
            new_batch["input_ids"].to(device),
            new_batch["attention_mask"].to(device),
            new_batch["labels"].to(device),
        )

        optimizer.zero_grad()

        # ✅ 核心优化 2：分别计算 Loss，保持逻辑独立
        old_loss = F.cross_entropy(
            model(old_ids, attention_mask=old_mask).logits, old_lbls
        )
        new_loss = F.cross_entropy(
            model(new_ids, attention_mask=new_mask).logits, new_lbls
        )

        # 混合 Loss
        loss = alpha * new_loss + (1 - alpha) * old_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / steps_per_epoch
    print(f"第 {epoch+1} 轮平均损失: {avg_loss:.4f}")