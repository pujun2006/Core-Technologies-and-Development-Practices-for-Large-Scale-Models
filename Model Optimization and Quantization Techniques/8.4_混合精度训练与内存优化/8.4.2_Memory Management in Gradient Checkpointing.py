import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


# ==========================================
# 1. 定义包含完整 Linear + Activation 的模块
# ==========================================
class NNBlock(nn.Module):

    def __init__(self, dim):
        super(NNBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


# ==========================================
# 2. 支持灵活开关 Checkpoint 的深度神经网络
# ==========================================
class DeepNN(nn.Module):

    def __init__(self, use_checkpointing=True):
        super(DeepNN, self).__init__()
        self.use_checkpointing = use_checkpointing

        # 构建深度网络层
        self.block1 = NNBlock(512)
        self.block2 = NNBlock(512)
        self.block3 = NNBlock(512)
        self.block4 = NNBlock(512)
        self.block5 = NNBlock(512)
        self.fc_out = nn.Linear(512, 10)

    def _run_block(self, block, x):
        # 使用现代 PyTorch 推荐的 use_reentrant=False 规范
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        x = self._run_block(self.block1, x)
        x = self._run_block(self.block2, x)
        x = self._run_block(self.block3, x)
        x = self._run_block(self.block4, x)
        x = self._run_block(self.block5, x)
        return self.fc_out(x)


# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前运行设备: {device}")

# 模拟数据 (适当增大 Batch Size 以便于在 GPU 上观察明显的显存差异)
input_data = torch.rand(256, 512, device=device)
labels = torch.randint(0, 10, (256,), device=device)
criterion = nn.CrossEntropyLoss()


# ==========================================
# 3. 训练函数定义
# ==========================================
def train_model(model, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# ==========================================
# 4. 显存峰值测量测试函数
# ==========================================
def measure_peak_memory(use_ckpt):
    # 初始化对应配置的模型与优化器
    model = DeepNN(use_checkpointing=use_ckpt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, labels)
    loss.backward()

    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        return mem_mb
    return 0.0


# ==========================================
# 5. 主程序执行
# ==========================================
if __name__ == "__main__":
    # 5.1 执行标准训练
    print("--- 1. 开始带有 Checkpointing 的训练 ---")
    model = DeepNN(use_checkpointing=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, optimizer, num_epochs=3)

    # 5.2 显存对比测试
    print("\n--- 2. 显存占用对比测试 (Peak Memory) ---")
    if device == "cuda":
        mem_no_ckpt = measure_peak_memory(use_ckpt=False)
        mem_with_ckpt = measure_peak_memory(use_ckpt=True)

        print(f"未开启 Checkpoint 最大显存占用: {mem_no_ckpt:.2f} MB")
        print(f"已开启 Checkpoint 最大显存占用: {mem_with_ckpt:.2f} MB")
        saved_ratio = ((mem_no_ckpt - mem_with_ckpt) / mem_no_ckpt) * 100
        print(f"显存优化比例: {saved_ratio:.2f}%")
    else:
        print("当前为 CPU 运行环境，无法进行 CUDA 显存峰值统计。")