import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 检查可用 GPU 数量
# ==========================================
gpu_count = torch.cuda.device_count()
print(f"当前可用 GPU 数量: {gpu_count}")

# ------------------------------------------
# Part 1: 数据并行 (DataParallel)
# ------------------------------------------
print("\n--- 1. 数据并行 (DataParallel) 演示 ---")


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dp_model = SimpleModel()

# 如果有多个 GPU，使用 DataParallel 包装
if gpu_count > 1:
    dp_model = nn.DataParallel(dp_model)

dp_model.to(device)

input_data_dp = torch.randn(64, 1024, device=device)
labels_dp = torch.randint(0, 10, (64,), device=device)

criterion_dp = nn.CrossEntropyLoss()
optimizer_dp = optim.Adam(dp_model.parameters(), lr=1e-3)

optimizer_dp.zero_grad()
outputs_dp = dp_model(input_data_dp)
loss_dp = criterion_dp(outputs_dp, labels_dp)
loss_dp.backward()
optimizer_dp.step()

print("数据并行处理的损失:", loss_dp.item())


# ------------------------------------------
# Part 2: 流水线模型并行 (Pipeline Parallelism)
# ------------------------------------------
print("\n--- 2. 模型并行 (Model Parallelism) 演示 ---")


class LargeModel(nn.Module):

    def __init__(self, dev0, dev1):
        super(LargeModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1

        # 将不同层分配到指定设备
        self.fc1 = nn.Linear(1024, 512).to(self.dev0)
        self.fc2 = nn.Linear(512, 256).to(self.dev1)
        self.fc3 = nn.Linear(256, 128).to(self.dev0)
        self.fc4 = nn.Linear(128, 10).to(self.dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = torch.relu(self.fc1(x))

        x = x.to(self.dev1)
        x = torch.relu(self.fc2(x))

        x = x.to(self.dev0)
        x = torch.relu(self.fc3(x))

        x = x.to(self.dev1)
        return self.fc4(x)


# 动态分配设备：若 GPU 少于 2 个则降级回单设备或 CPU 避免 Crash
if gpu_count >= 2:
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
else:
    dev0 = device
    dev1 = device

large_model = LargeModel(dev0, dev1)

input_data_mp = torch.randn(64, 1024, device=dev0)
labels_mp = torch.randint(0, 10, (64,), device=dev1)

criterion_mp = nn.CrossEntropyLoss()
optimizer_mp = optim.Adam(large_model.parameters(), lr=1e-3)

optimizer_mp.zero_grad()
outputs_mp = large_model(input_data_mp)
loss_mp = criterion_mp(outputs_mp, labels_mp)
loss_mp.backward()
optimizer_mp.step()

print("模型并行处理的损失:", loss_mp.item())