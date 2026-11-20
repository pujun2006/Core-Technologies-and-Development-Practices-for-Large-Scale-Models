import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# ==========================================
# 1. 基础模型定义
# ==========================================
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


# ==========================================
# Part 1: DataParallel (单进程多卡) 演示
# ==========================================
def run_dataparallel_demo():
    print("--- 1. DataParallel 训练演示 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # 仅在多 GPU 时使用 DP
    model.to(device)

    input_data = torch.randn(64, 1024, device=device)
    labels = torch.randint(0, 10, (64,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    outputs = model(input_data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"DataParallel 训练的损失: {loss.item():.4f}\n")


# ==========================================
# Part 2: DistributedDataParallel (多进程多卡) 演示
# ==========================================
def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 显式设置当前进程使用的 CUDA 设备，防止 GPU 0 显存溢出
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_ddp_process(rank, world_size):
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model = SimpleModel().to(device)
    # 使用 DDP 包装模型
    model = DDP(model, device_ids=[rank])

    input_data = torch.randn(64, 1024, device=device)
    labels = torch.randint(0, 10, (64,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    outputs = model(input_data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"[Rank {rank}] DistributedDataParallel 训练的损失: {loss.item():.4f}"
    )

    dist.destroy_process_group()


# ==========================================
# 主程序入口 (Python 多进程必须包含 if __name__ == '__main__')
# ==========================================
if __name__ == "__main__":
    # 1. 运行 DataParallel 测试
    run_dataparallel_demo()

    # 2. 运行 DistributedDataParallel 测试
    gpu_count = torch.cuda.device_count()
    print("--- 2. DistributedDataParallel (DDP) 训练演示 ---")

    if gpu_count >= 2:
        print(f"检测到 {gpu_count} 张 GPU，启动多进程 DDP 训练...")
        torch.multiprocessing.spawn(
            train_ddp_process, args=(gpu_count,), nprocs=gpu_count, join=True
        )
    else:
        print(
            f"当前可用 GPU 数量为 {gpu_count}，DDP 建议在 2 张及以上 GPU 环境下运行测试。"
        )