import torch
import torch.nn as nn
import torch.profiler as profiler


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

        # 【关键修正】：28x28 图像无池化层时尺寸不变，维度应为 32 * 28 * 28 = 25088
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 初始化模型与测试数据
model = SimpleCNN().to("cpu")
model.eval()  # 建议在测试推理时切换为评估模式
input_data = torch.rand(16, 1, 28, 28)

# ==========================================
# 2. CPU 性能剖析 (Profiler)
# ==========================================
print("--- 1. CPU 性能分析中... ---")
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    record_shapes=True,  # 记录算子的输入张量形状 (Tensor Shapes)
    profile_memory=True,  # 记录 CPU 内存分配与释放
) as prof_cpu:
    with profiler.record_function("model_inference_cpu"):
        output = model(input_data)

# 打印 CPU 耗时前 10 名的算子
print(
    prof_cpu.key_averages().table(sort_by="cpu_time_total", row_limit=10)
)

# ==========================================
# 3. GPU 性能剖析 (需要 CUDA 设备支持)
# ==========================================
if torch.cuda.is_available():
    print("\n--- 2. GPU (CUDA) 性能分析中... ---")
    device = "cuda"
    model.to(device)
    input_data = input_data.to(device)

    # 预热 GPU (Warmup)，避免首次 GPU 内存初始化影响分析数据
    with torch.no_grad():
        _ = model(input_data)
    torch.cuda.synchronize()

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,  # 记录 GPU 显存 (CUDA Memory) 的分配细节
        with_stack=False,  # 设置为 True 可追踪算子对应的 Python 代码行
    ) as prof_gpu:
        with profiler.record_function("model_inference_gpu"):
            output = model(input_data)
            torch.cuda.synchronize()  # 确保所有 CUDA Kernel 均已执行完毕

    # 按 CUDA 耗时排序输出
    print(
        prof_gpu.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        )
    )

    # 导出 Chrome Trace 追踪文件 (可在 Chrome 浏览器访问 chrome://tracing 或 ui.perfetto.dev 打开)
    prof_gpu.export_chrome_trace("trace_gpu.json")
    print("\n性能追踪日志已导出至 'trace_gpu.json'！")
else:
    print("\n当前系统未检测到可用 CUDA GPU，跳过 GPU 性能剖析。")