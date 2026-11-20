import torch
import torch.nn as nn
import torch.optim as optim


# 1. 定义模型
class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 2. 设备与数据初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)

# 模拟数据：总 Batch Size = 16
input_data = torch.randn(16, 1024, device=device)
labels = torch.randint(0, 10, (16,), device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. 梯度累积配置
accumulation_steps = 4  # 相当于实际按 4*4=16 的等效 Batch Size 更新一次参数
micro_batch_size = 4  # 每次前向传播的微批次大小

# 4. 梯度累积训练循环
for epoch in range(2):
    model.train()
    optimizer.zero_grad()  # 在每个 Epoch 或累积序列开始前清空一次梯度

    for i in range(accumulation_steps):
        # 切片提取微批次 (Micro Batch)
        start_idx = i * micro_batch_size
        end_idx = (i + 1) * micro_batch_size

        input_batch = input_data[start_idx:end_idx]
        labels_batch = labels[start_idx:end_idx]

        # 前向传播
        outputs = model(input_batch)

        # 【核心步骤】：损失归一化，防止梯度按累积步数放大
        loss = criterion(outputs, labels_batch) / accumulation_steps

        # 反向传播：PyTorch 会在 param.grad 上自动做累加 (+=)
        loss.backward()

        # 还原打印实际的未缩放 Loss 值，便于日志观察
        actual_loss = loss.item() * accumulation_steps
        print(
            f"Epoch [{epoch+1}], Step [{i+1}/{accumulation_steps}], Micro-Batch Loss: {actual_loss:.4f}"
        )

    # 累积完成后，一次性更新参数并清空梯度
    optimizer.step()
    optimizer.zero_grad()

print("\n梯度累积训练成功完成！")