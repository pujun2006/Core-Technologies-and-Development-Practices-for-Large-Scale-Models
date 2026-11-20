import time
import torch
import torch.nn as nn
# 使用 PyTorch 推荐的统一架构量化模块
import torch.ao.quantization as quantization


# ==========================================
# 1. 支持静态量化的模型（必须包含 QuantStub / DeQuantStub）
# ==========================================
class SimpleFCN(nn.Module):

    def __init__(self):
        super(SimpleFCN, self).__init__()
        # 1.1 静态量化必备的输入/输出转换桩
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 显式加入量化与反量化节点
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


# 测试推理时间与延迟
def test_inference(model, input_data, description=""):
    # 预热 CPU 缓存
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(input_data)
    end = time.time()
    print(f"{description} 推理时间: {(end - start):.4f} 秒")


# ==========================================
# 2. 运行量化测试
# ==========================================
if __name__ == "__main__":
    # 初始化模型并移至 CPU (量化主要在 CPU/x86/ARM 上运行加速)
    model = SimpleFCN().to("cpu")
    model.eval()

    input_data = torch.randn(1024, 784)

    # ------------------------------------------
    # A. 原始 Float32 模型推理
    # ------------------------------------------
    test_inference(model, input_data, "1. 原始 Float32 模型")

    # ------------------------------------------
    # B. 动态量化 (Dynamic Quantization)
    # 动态量化不需要 QuantStub，也不需要校准，仅量化 Weight，Activation 在运行时动态量化
    # ------------------------------------------
    dynamic_model = quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    test_inference(dynamic_model, input_data, "2. 动态量化模型")

    # ------------------------------------------
    # C. 静态量化 (Static Quantization)
    # 步骤：设置 Config -> 准备 (Prepare) -> 校准 (Calibrate) -> 转换 (Convert)
    # ------------------------------------------
    static_model = SimpleFCN().to("cpu")
    static_model.eval()

    # 1) 指定 backend 配置（x86 架构选 fbgemm，ARM/移动端选 qnnpack）
    static_model.qconfig = quantization.get_default_qconfig("fbgemm")

    # 2) 准备阶段：插入观察器 (Observer)
    prepared_model = quantization.prepare(static_model)

    # 3) 校准阶段 (Calibration)：必须用代表性数据运行前向传播，收集激活值的分布数据！
    print("\n[正在进行静态量化数据校准...]")
    with torch.no_grad():
        for _ in range(10):
            calibration_batch = torch.randn(128, 784)
            prepared_model(calibration_batch)

    # 4) 转换阶段：将 Observer 替换为真实的 INT8 算子
    quantized_model = quantization.convert(prepared_model)

    test_inference(quantized_model, input_data, "3. 静态量化模型")