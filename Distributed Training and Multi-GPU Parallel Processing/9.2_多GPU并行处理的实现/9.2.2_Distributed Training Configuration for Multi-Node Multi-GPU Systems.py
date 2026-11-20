# 单机模拟 4 进程分布式训练

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


# 1. 修正后的初始化函数
def setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # 判断硬件环境
    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"

    # 初始化分布式进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)


# 2. 简单模型定义
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


# 3. 模拟训练函数
# 【注意参数顺序】：spawn 自动传入的 index 会赋值给第一个参数 fn_rank
def train(fn_rank, world_size, master_addr, master_port):
    rank = fn_rank  # fn_rank 就是当前进程的 Rank (0, 1, 2, 3)

    setup(rank, world_size, master_addr, master_port)

    # 设备分配逻辑
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # 映射到可用 GPU 上（若只有 1 张 GPU，所有 Rank 会映射到 cuda:0 上进行模拟）
        gpu_id = rank % gpu_count
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        model = SimpleModel().to(device)
        # 单卡模拟多进程时，将 device_ids 指定为当前进程对应的显卡
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu_id]
        )
    else:
        # CPU 模拟模式
        device = torch.device("cpu")
        model = SimpleModel().to(device)
        model = nn.parallel.DistributedDataParallel(model)

    # 构造模拟数据
    input_data = torch.randn(64, 1024, device=device)
    labels = torch.randint(0, 10, (64,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 前向传播与反向传播 (DDP 自动完成 4 个进程的梯度同步)
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(
        f"[Rank {rank}/{world_size}] 在设备 {device} 上完成训练，Loss: {loss.item():.4f}"
    )

    # 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    # 配置本地回环地址进行单机模拟
    master_addr = "127.0.0.1"  # 本地模拟使用 127.0.0.1
    master_port = "12355"
    world_size = 4  # 模拟 4 个分布式进程

    print(f"开始在单机上模拟启动 {world_size} 个进程进行 DDP 训练...")

    # 【关键修正】：mp.spawn 调用的函数签名必须是 train(rank, *args)
    # args 传给 train 中 rank 之后的参数，即 (world_size, master_addr, master_port)
    torch.multiprocessing.spawn(
        train,
        args=(world_size, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )