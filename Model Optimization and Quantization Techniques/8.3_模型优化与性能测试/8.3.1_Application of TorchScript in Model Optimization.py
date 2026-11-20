import time
import torch
import torch.nn as nn


# 修正后的 SimpleCNN 模型
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 此时 32 * 7 * 7 正确匹配！
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ------------------------------------------
# 测试与 JIT 导出主程序
# ------------------------------------------
if __name__ == "__main__":
    # 1. 模型初始化与评估模式设置
    model = SimpleCNN().to("cpu")
    model.eval()  # 【最佳实践】TorchScript 转换前建议显式切换为 eval() 模式

    input_data = torch.rand(16, 1, 28, 28)

    # 推理性能测试函数
    def test_inference(model_obj, input_tensor, description=""):
        # 预热 CPU 缓存
        with torch.no_grad():
            for _ in range(10):
                _ = model_obj(input_tensor)

        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model_obj(input_tensor)
        end = time.time()
        print(f"[{description}] 推理时间: {(end - start):.4f} 秒")
        return output

    # 2. 原始模型推理
    test_inference(model, input_data, "1. 原始 PyTorch 模型")

    # 3. TorchScript 追踪法 (Trace)
    traced_model = torch.jit.trace(model, input_data)
    test_inference(traced_model, input_data, "2. TorchScript 追踪法 (Trace)")

    # 4. TorchScript 脚本法 (Script)
    scripted_model = torch.jit.script(model)
    test_inference(scripted_model, input_data, "3. TorchScript 脚本法 (Script)")

    # 5. 保存并加载 TorchScript 模型
    traced_model.save("traced_model.pt")
    scripted_model.save("scripted_model.pt")

    loaded_traced_model = torch.jit.load("traced_model.pt")
    loaded_scripted_model = torch.jit.load("scripted_model.pt")

    test_inference(
        loaded_traced_model, input_data, "4. 加载磁盘上的 Trace 模型"
    )
    test_inference(
        loaded_scripted_model, input_data, "5. 加载磁盘上的 Script 模型"
    )