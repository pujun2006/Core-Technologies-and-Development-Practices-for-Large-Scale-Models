import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. 修正维度后的 SimpleCNN 模型
# ==========================================
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 【关键修正】：28x28 图像无池化时尺寸保持 28x28，维度为 32 * 28 * 28 = 25088
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前运行设备: {device}")

# 初始化模型、优化器和损失函数
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 【最新 API 规范】：使用 torch.amp.GradScaler('cuda')
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

# 模拟训练数据
input_data = torch.rand(16, 1, 28, 28).to(device)
labels = torch.randint(0, 10, (16,)).to(device)


# ==========================================
# 2. AMP 混合精度训练函数
# ==========================================
def train_amp(model, optimizer, criterion, scaler, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 【最新 API 规范】：显式指定 device_type='cuda'
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            outputs = model(input_data)
            loss = criterion(outputs, labels)

        # 梯度缩放与优化器更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


print("\n--- 1. 开始自动混合精度 (AMP) 训练 ---")
train_amp(model, optimizer, criterion, scaler)


# ==========================================
# 3. AMP 混合精度推理测试函数
# ==========================================
def test_inference_amp(model, input_data):
    model.eval()
    with torch.no_grad():
        # 推理时开启 autocast 可降低显存开销并提升计算吞吐
        with torch.amp.autocast(
            device_type="cuda", enabled=(device == "cuda")
        ):
            output = model(input_data)
            print("推理输出示例 (前 10 维):\n", output[0, :10])


print("\n--- 2. AMP 推理测试 ---")
test_inference_amp(model, input_data)