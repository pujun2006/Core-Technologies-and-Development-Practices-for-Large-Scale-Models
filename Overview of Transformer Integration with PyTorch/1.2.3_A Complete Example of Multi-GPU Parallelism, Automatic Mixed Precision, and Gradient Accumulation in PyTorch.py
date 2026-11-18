import torch
import torch.nn as nn
import torch.optim as optim

# 1. 检查可用的设备，并选择 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前计算设备:", device)

device_type = "cuda" if torch.cuda.is_available() else "cpu"


# 2. 定义一个更复杂的神经网络
class ComplexNet(nn.Module):

  def __init__(self):
    super(ComplexNet, self).__init__()
    self.fc1 = nn.Linear(512, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 10)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = self.fc4(x)
    return x


# 3. 实例化模型，使用 DataParallel 进行多 GPU 支持
model = ComplexNet().to(device)
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
  print(f"使用多 GPU 进行计算，GPU 数量: {torch.cuda.device_count()}")

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 初始化新版统一规范的 GradScaler (支持 'cuda' 或 'cpu')
scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))

# 6. 准备总批次数据（模拟大批次）
batch_size = 128
accumulation_steps = 4  # 梯度累积步数
micro_batch_size = batch_size // accumulation_steps  # 每个小批次的大小 = 32

input_data = torch.rand((batch_size, 512)).to(device)
target = torch.randint(0, 10, (batch_size,)).to(device)  # 随机生成目标标签

# 7. 开始训练流程（梯度累积 + 混合精度）
model.train()
optimizer.zero_grad()  # 训练开始前清空梯度

print("\n开始执行梯度累积训练...")
for i in range(accumulation_steps):
  # 计算当前 micro-batch 的切片索引
  start_idx = i * micro_batch_size
  end_idx = (i + 1) * micro_batch_size

  micro_input = input_data[start_idx:end_idx]
  micro_target = target[start_idx:end_idx]

  # 使用新版 torch.amp.autocast 进行混合精度前向传播
  with torch.amp.autocast(
      device_type=device_type, enabled=(device_type == "cuda")
  ):
    outputs = model(micro_input)
    loss = criterion(outputs, micro_target)
    loss = loss / accumulation_steps  # 关键：对损失进行归一化，平衡梯度量级

  # 缩放 Loss 并进行反向传播（累积梯度）
  scaler.scale(loss).backward()
  print(
      f"累积步数 [{i + 1}/{accumulation_steps}], 当前小批次损失: {loss.item() * accumulation_steps:.4f}"
  )

# 8. 累积完成后，统一更新模型权重
scaler.step(optimizer)
scaler.update()
print("权重更新完成！")

# 9. 模型测试与推理
with torch.no_grad():
  model.eval()
  test_data = torch.rand((1, 512)).to(device)
  with torch.amp.autocast(
      device_type=device_type, enabled=(device_type == "cuda")
  ):
    output = model(test_data)
  print("\n模型测试输出形状:", output.shape)
  print("模型测试输出预测结果:\n", output)