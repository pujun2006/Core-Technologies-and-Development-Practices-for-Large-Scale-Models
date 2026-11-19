import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# 1. 定义模型名称并加载预训练分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 2. 加载 CSV 文件并进行数据清洗（安全加固）
def load_data(file_path):
    """加载 CSV 文件，清理缺失值与首尾空格，并重置索引"""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["text", "label"])  # 明确指定关键列
    df["text"] = df["text"].astype(str).str.strip()  # 确保字符串格式并去空格
    df = df.reset_index(drop=True)  # 重置索引，防止断档
    return df


# 3. 优化后的 Dataset：只截断，不强行 Padding
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

        # 仅在此处做分词与截断，Padding 留给 DataLoader 动态处理
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # 返回纯 Python 列表，方便 DataCollator 快速处理
        )

        inputs["labels"] = int(label)
        return inputs


# 4. 加载数据
file_path = "领域数据集.csv"
# 兼容性预检查（如果文件不存在，生成临时模拟数据避免报错）
try:
    df = load_data(file_path)
except FileNotFoundError:
    df = pd.DataFrame(
        {
            "text": [
                "自然语言处理技术发展迅速。",
                "深度学习在文本分类中表现优异。",
                "这是一个短句。",
            ],
            "label": [1, 1, 0],
        }
    )

texts = df["text"].tolist()
labels = df["label"].tolist()

# 5. 创建数据集与带有动态 Padding 的 DataLoader
dataset = CustomTextDataset(texts, labels, tokenizer, max_length=128)

# 使用 Hugging Face 官方的高性能动态填充器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator,  # 按 Batch 内部最长序列动态对齐
)

# 6. 验证输出 Batch 的维度与结构
for batch in dataloader:
    print("=== 动态 Padding 验证 ===")
    print("input_ids 形状:", batch["input_ids"].shape)
    print("attention_mask 形状:", batch["attention_mask"].shape)
    print("labels 形状与数据:", batch["labels"])
    break

import torch
import torch.nn as nn
from torch.optim import AdamW  # ✅ 使用 PyTorch 原生 AdamW
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# 1. 设备配置与模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 2. 结合层冻结（冻结 Embedding 与 前 6 层 Encoder，节约显存与计算）
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

model.to(device)

# 3. 优化器仅过滤未冻结（requires_grad=True）的参数
initial_lr = 2e-5
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr
)

# 4. 配置 Learning Rate Scheduler
num_epochs = 3
total_steps = num_epochs * len(dataloader)  # dataloader 需已正确定义
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=total_steps
)

# 5. 规范的标准训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        # 批量数据送入设备 (GPU/CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播：传入 labels 时，模型会自动计算并返回 Loss
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss  # ✅ 直接使用模型内建计算出的 Loss

        # 反向传播
        loss.backward()

        # ✅ 关键优化：梯度裁剪，防止梯度爆炸 (Gradient Clipping)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )

        # 参数更新与学习率调整
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch+1}/{num_epochs} - 平均 Loss: {avg_loss:.4f} | 当前学习率: {current_lr:.8f}"
    )