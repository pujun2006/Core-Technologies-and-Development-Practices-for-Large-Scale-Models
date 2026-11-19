import torch
import torch.nn.functional as F
from torch.optim import AdamW  # ✅ 使用 PyTorch 原生 AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 定义模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 2. 模拟数据
old_texts = ["旧数据样本1", "旧数据样本2", "旧数据样本3"]
new_texts = ["新数据样本1", "新数据样本2"]
old_labels = [0, 1, 0]
new_labels = [1, 2]

combined_texts = old_texts + new_texts
combined_labels = old_labels + new_labels


# 3. CustomDataset 与 动态 Collator
class CustomTextDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": str(self.texts[idx]), "labels": self.labels[idx]}


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


combined_dataset = CustomTextDataset(combined_texts, combined_labels)

# 4. 加权采样器设置（旧数据 0.5，新数据 1.0）
combined_weights = [
    0.5 if i < len(old_texts) else 1.0 for i in range(len(combined_labels))
]
sampler = WeightedRandomSampler(
    combined_weights, num_samples=len(combined_weights), replacement=True
)

dataloader = DataLoader(
    combined_dataset,
    batch_size=2,
    sampler=sampler,
    collate_fn=collate_fn,  # 注意：使用了 sampler 绝不能写 shuffle=True
)

# 5. 设备准备与初始参数快照
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 修复 1：将快照移动到对应 device，并使用 detach() 切断计算图
initial_params = {
    name: param.clone().detach().to(device)
    for name, param in model.named_parameters()
    if param.requires_grad
}

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 训练配置
num_epochs = 3
beta = 0.1  # 参数偏差惩罚系数 (L2 Constraint)

# 6. 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # 计算交叉熵 Loss（已通过 WeightedRandomSampler 完成新旧权重平衡）
        outputs = model(input_ids, attention_mask=attention_mask)
        data_loss = F.cross_entropy(outputs.logits, labels)

        # ✅ 修复 2：高效计算二范数平方惩罚，不产生 Device 冲突和显存泄露
        constraint_loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in initial_params:
                constraint_loss += torch.sum((param - initial_params[name]) ** 2)

        # ✅ 修复 3：总损失 = 数据交叉熵损失 + 权重锚定惩罚
        total_loss_value = data_loss + beta * constraint_loss

        total_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_loss_value.item()

    avg_loss = total_loss / len(dataloader)
    print(f"第 {epoch+1} 轮平均损失: {avg_loss:.4f}")
