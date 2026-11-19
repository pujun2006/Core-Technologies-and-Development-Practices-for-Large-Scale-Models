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

# 2. 准备数据
old_texts = ["旧数据样本1", "旧数据样本2", "旧数据样本3"]
new_texts = ["新数据样本1", "新数据样本2"]
old_labels = [0, 1, 0]
new_labels = [1, 2]


# 3. 自定义 Dataset 与 Collator（动态 Padding）
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


old_dataset = CustomTextDataset(old_texts, old_labels)
new_dataset = CustomTextDataset(new_texts, new_labels)

old_dataloader = DataLoader(
    old_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)
new_dataloader = DataLoader(
    new_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

# 4. 模型迁移与设备准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 修复 1：保存初始参数快照，并确保 detach() 且与 model 处在同一 device 上
initial_params = {
    name: param.clone().detach().to(device)
    for name, param in model.named_parameters()
    if param.requires_grad
}

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 3
alpha = 0.5  # 新旧数据 Loss 平衡系数
beta = 0.1  # 参数偏差惩罚系数 (L2 Regularization Strength)

# 5. 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    # ✅ 修复 2：使用 cycle 防止旧数据丢弃
    steps_per_epoch = len(old_dataloader)
    new_dataloader_iter = itertools.cycle(new_dataloader)

    for old_batch in old_dataloader:
        new_batch = next(new_dataloader_iter)

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

        # 1. 新旧数据的交叉熵 Loss
        old_loss = F.cross_entropy(
            model(old_ids, attention_mask=old_mask).logits, old_lbls
        )
        new_loss = F.cross_entropy(
            model(new_ids, attention_mask=new_mask).logits, new_lbls
        )

        # ✅ 修复 3：正确且高效地计算参数偏移 L2 惩罚项
        constraint_loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in initial_params:
                # 计算与初始权重的二范数平方和
                constraint_loss += torch.sum((param - initial_params[name]) ** 2)

        # 混合总损失
        loss = alpha * new_loss + (1 - alpha) * old_loss + beta * constraint_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / steps_per_epoch
    print(f"第 {epoch+1} 轮平均损失: {avg_loss:.4f}")