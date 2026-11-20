import time
import torch
import torch.ao.quantization as quantization
import torch.nn as nn
import torch.optim as optim


# 1. 网络模型
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x


# 2. 彻底解决 Fusion 报错的融合函数
def fuse_cnn_model(model):
    modules_to_fuse = [
        ["conv1", "bn1", "relu1"],
        ["conv2", "bn2", "relu2"],
        ["fc1", "relu3"],
    ]

    # 递归将所有子模块强制设为 train() 模式
    model.train()
    for m in model.modules():
        m.training = True

    # 如果存在 fuse_modules_qat 则优先直接调用
    if hasattr(quantization, "fuse_modules_qat"):
        return quantization.fuse_modules_qat(model, modules_to_fuse)

    return quantization.fuse_modules(model, modules_to_fuse)


# 3. 运行主程序
if __name__ == "__main__":
    device = "cpu"

    model = SimpleCNN().to(device)

    # 算子融合 (此时不会再报 AssertionError)
    model = fuse_cnn_model(model)

    # 准备 QAT 训练
    model.qconfig = quantization.get_default_qat_qconfig("fbgemm")
    prepared_model = quantization.prepare_qat(model, inplace=False)

    optimizer = optim.Adam(prepared_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    input_data = torch.rand(16, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (16,)).to(device)

    num_epochs = 10
    print("--- 1. 开始 QAT 训练 ---")
    for epoch in range(num_epochs):
        prepared_model.train()

        if epoch >= num_epochs - 3:
            prepared_model.apply(quantization.disable_observer)
            prepared_model.apply(nn.intrinsic.qat.freeze_bn_stats)

        optimizer.zero_grad()
        outputs = prepared_model(input_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # 转换为纯 INT8 模型
    prepared_model.eval()
    quantized_model = quantization.convert(
        prepared_model.to("cpu"), inplace=False
    )

    print("\n--- 2. 量化模型推理测试 ---")
    with torch.no_grad():
        test_input = torch.rand(1, 1, 28, 28)
        output = quantized_model(test_input)
        print("输出 Logits 成功:\n", output[0, :10])