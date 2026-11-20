import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer


# 1. 定义标准的蒸馏损失函数 (Hinton Distillation Loss)
class DistillationLoss(nn.Module):

    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # 蒸馏损失：Soft Target 匹配 (注意带上 T^2 缩放因子)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        distillation_loss = self.kl_div(soft_student, soft_teacher) * (
            self.temperature**2
        )

        # 交叉熵损失：Hard Target 学习
        ce_loss = self.ce_loss(student_logits, labels)

        # 动态加权求和
        return self.alpha * distillation_loss + (1.0 - self.alpha) * ce_loss


# 2. 学生模型训练函数
def train_student_model(
    teacher_model,
    student_model,
    dataloader,
    optimizer,
    distillation_loss_fn,
    device="cuda",
):
    student_model.train()
    teacher_model.eval()

    total_loss = 0.0

    for batch in dataloader:
        # 正确解包张量，并统一移动到目标设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 教师模型前向传播（无梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits

        # 学生模型前向传播
        student_outputs = student_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_logits = student_outputs.logits

        # 计算加权蒸馏损失
        loss = distillation_loss_fn(student_logits, teacher_logits, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# 3. 动态 Alpha 调度策略 (通常建议随训练推进逐渐增加 hard label 比例或增加 soft label 比例)
def dynamic_alpha_schedule(epoch, total_epochs, start_alpha=0.7, end_alpha=0.3):
    return start_alpha + (end_alpha - start_alpha) * (epoch / total_epochs)


# ==========================================
# 主运行程序
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型并正确移动到设备 (Device)
    print(f"正在使用设备: {device}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    teacher_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    student_model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    ).to(device)

    # 冻结教师模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 构建标准数据集格式（包含 attention_mask）
    def create_dummy_dataset(tokenizer, num_samples=16):
        data = []
        texts = ["Sample text for distillation test."] * num_samples
        labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        for i in range(num_samples):
            data.append(
                {
                    "input_ids": encoded["input_ids"][i],
                    "attention_mask": encoded["attention_mask"][i],
                    "labels": torch.tensor(labels[i]),
                }
            )
        return data

    dummy_data = create_dummy_dataset(tokenizer)
    dataloader = DataLoader(dummy_data, batch_size=4, shuffle=True)

    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)
    distillation_loss_fn = DistillationLoss(temperature=2.0, alpha=0.7)

    # 开始动态 Alpha 循环训练
    total_epochs = 5
    print("\n--- 开始知识蒸馏训练 ---")
    for epoch in range(total_epochs):
        current_alpha = dynamic_alpha_schedule(epoch, total_epochs)
        distillation_loss_fn.alpha = current_alpha

        avg_loss = train_student_model(
            teacher_model,
            student_model,
            dataloader,
            optimizer,
            distillation_loss_fn,
            device=device,
        )

        print(
            f"Epoch {epoch+1}/{total_epochs} | Alpha: {current_alpha:.2f} | 平均损失: {avg_loss:.4f}"
        )