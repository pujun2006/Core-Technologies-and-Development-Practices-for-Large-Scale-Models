import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# 1. 初始化分布式环境
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 后端选择：GPU 环境推荐 "nccl"，仅 CPU 环境使用 "gloo"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# 2. 模型定义
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


# 3. 分布式训练主函数
def train(rank, world_size):
    setup(rank, world_size)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        # GPU 模式必须指定 device_ids
        model = DDP(model, device_ids=[rank])
    else:
        device = torch.device("cpu")
        model = SimpleModel().to(device)
        # CPU 模式切勿传递 device_ids
        model = DDP(model)

    # 构造模拟数据（实际训练中建议使用 DistributedSampler 加载真实 Dataset）
    input_data = torch.randn(64, 1024, device=device)
    labels = torch.randint(0, 10, (64,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 标准 DDP 前向与反向传播
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, labels)

    # 【核心】：loss.backward() 会自动在后台执行梯度的分布式 All-Reduce 平均！
    # 绝对不需要任何手动的 dist.all_reduce() 操作！
    loss.backward()

    # 参数更新
    optimizer.step()

    print(f"[Rank {rank}] 训练完成，Loss: {loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    # 检测可用 GPU 数，若小于 2 则默认使用 2 个进程模拟（或切换为 CPU）
    world_size = (
        torch.cuda.device_count() if torch.cuda.device_count() >= 2 else 2
    )
    print(f"启动 {world_size} 个进程进行 DDP 训练...")

    torch.multiprocessing.spawn(
        train, args=(world_size,), nprocs=world_size, join=True
    )