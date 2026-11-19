import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# 1. 定义模型名称并加载分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 准备示范数据（若实际项目中有 CSV 文件，可用 pd.read_csv 加载）
sample_data = {
    "text": [
        "自然语言处理是一门研究计算机如何理解和生成人类语言的学科。",
        "在实际应用中，文本数据通常具有不确定的长度。",
        "通过使用填充和截断技术，确保每个批次内的序列长度一致，提升模型的处理效率。",
        "深度学习在文本分类任务中取得了显著的成果。",
        "使用 PyTorch 构建端到端的 NLP 管道非常高效。",
    ],
    "label": [1, 1, 0, 1, 0],
}
df = pd.DataFrame(sample_data)


# 3. 定义自定义 Dataset 类（仅做截断，不在此处强行 Padding）
class CustomTextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # 返回纯 Python 列表，方便 DataCollator 快速处理
        )
        inputs["labels"] = int(label)
        return inputs


# 4. 实例化 Dataset
dataset = CustomTextDataset(
    texts=df["text"].tolist(),
    labels=df["label"].tolist(),
    tokenizer=tokenizer,
    max_length=128,
)

# 5. 使用 Hugging Face 官方的高性能动态填充器 (DataCollatorWithPadding)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# 6. 定义 dataloader (供训练循环使用)
dataloader = DataLoader(
    dataset,
    batch_size=2,  # 批次大小
    shuffle=True,  # 训练集打乱顺序
    collate_fn=data_collator,  # 按 Batch 内最长句子动态填充
)

# 验证 dataloader 输出结构
print("✅ DataLoader 定义成功！示例 Batch 结构如下：")
for batch in dataloader:
    print("input_ids 形状:", batch["input_ids"].shape)
    print("attention_mask 形状:", batch["attention_mask"].shape)
    print("labels 形状与数据:", batch["labels"])
    break

import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 2. 初始冻结 Embedding 与 前 6 层 Encoder (0-5)
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

model.to(device)

num_epochs = 3
initial_lr = 2e-5


# 3. 辅助函数：根据当前 requires_grad 状态重新创建或更新优化器
def create_optimizer_and_scheduler(model, lr, total_steps):
    # 过滤出当前可训练的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )
    return optimizer, scheduler


# 初始化第 1 轮的优化器与调度器
total_steps = num_epochs * len(dataloader)
optimizer, scheduler = create_optimizer_and_scheduler(
    model, initial_lr, total_steps
)

# 4. 渐进式解冻训练循环
for epoch in range(num_epochs):
    # 在 epoch > 0 时安全解冻指定层
    if epoch > 0:
        unfreeze_index = 6 - epoch
        if unfreeze_index >= 0:
            print(
                f"\n🔓 [Epoch {epoch+1}] 解冻 Transformer Block Layer[{unfreeze_index}]..."
            )
            for param in model.bert.encoder.layer[unfreeze_index].parameters():
                param.requires_grad = True

            # ✅ 核心修复：修改完 requires_grad 后，必须重新生成优化器以包含新解冻的参数！
            # 计算剩余训练步数
            remaining_steps = (num_epochs - epoch) * len(dataloader)
            optimizer, scheduler = create_optimizer_and_scheduler(
                model, initial_lr, remaining_steps
            )

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        loss.backward()
        # 建议加上梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} 平均损失: {avg_loss:.4f}")