import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# 1. 修正后的初始化函数
def setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # 自动根据硬件环境选择通信后端
    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"

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


# 3. 分布式训练函数
# 【注意】：fn_rank 是 mp.spawn 自动注入的当前进程 ID (0 ~ world_size-1)
def train(fn_rank, world_size, master_addr, master_port):
    rank = fn_rank  # 明确标识 Rank

    setup(rank, world_size, master_addr, master_port)

    # 设备分配逻辑
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_id = rank % gpu_count

        # 必须显式设置当前线程的 GPU 设备，防止 GPU 0 显存溢出
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        model = SimpleModel().to(device)
        model = DDP(model, device_ids=[gpu_id])
    else:
        device = torch.device("cpu")
        model = SimpleModel().to(device)
        model = DDP(model)

    # 模拟数据
    input_data = torch.randn(16, 1024, device=device)
    labels = torch.randint(0, 10, (16,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(2):
        optimizer.zero_grad()

        outputs = model(input_data)
        loss = criterion(outputs, labels)

        loss.backward()  # DDP 自动执行 Ring-AllReduce 梯度同步
        optimizer.step()

        print(
            f"[Rank {rank}] Epoch [{epoch+1}/2], Loss: {loss.item():.4f} (设备: {device})"
        )

    dist.destroy_process_group()


# 4. 主程序入口
if __name__ == "__main__":
    # 单机本地模拟参数
    master_addr = "127.0.0.1"  # 本地回环 IP
    master_port = "12355"  # 空闲端口
    world_size = 4  # 单机模拟 4 个分布式进程

    print(f"开始在单机上模拟 {world_size} 个进程进行 DDP 训练...\n")

    # 【关键修正】：mp.spawn 会将进程索引作为第 1 个参数自动传给 train()
    # 这里的 args 只需要传入 train 函数中 rank 之后的参数：(world_size, master_addr, master_port)
    torch.multiprocessing.spawn(
        train,
        args=(world_size, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )