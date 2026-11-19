import torch
import torch.nn as nn
import torch.optim as optim

# 1. 检查GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. 定义网络
class SimpleNet(nn.Module):

  def __init__(self):
    super(SimpleNet, self).__init__()
    self.fc1 = nn.Linear(100, 50)
    self.fc2 = nn.Linear(50, 10)
    self.fc3 = nn.Linear(10, 1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x


# 3. 实例化模型并移动到设备
model = SimpleNet().to(device)

# 4. 【关键修正】如果有多张 GPU，先做 DataParallel 封装，再绑定优化器
if torch.cuda.device_count() > 1:
  print(f"使用多GPU进行计算，GPU数量: {torch.cuda.device_count()}")
  model = nn.DataParallel(model)

# 5. 此时再定义优化器，确保它能正确捕获当前模型（或 DataParallel 内部模块）的参数
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 6. 准备数据
input_data = torch.rand((64, 100), device=device)
target = torch.rand((64, 1), device=device)


# 7. 训练函数
def train_step(model, data, target, criterion, optimizer):
  model.train()
  optimizer.zero_grad()
  output = model(data)
  loss = criterion(output, target)
  loss.backward()
  optimizer.step()
  return loss.item()


# 8. 运行训练
loss_val = train_step(model, input_data, target, criterion, optimizer)
print("训练损失:", loss_val)
