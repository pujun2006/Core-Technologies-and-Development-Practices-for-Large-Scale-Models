# 动态图计算与自动微分机制
import torch

# 创建一个张量，设置requires_grad=True以便追踪梯度
x=torch.tensor([2.0, 3.0], requires_grad=True)
# 定义一个简单的计算图 y=x1^2+x2^3
y=x[0]**2+x[1]**3
# 打印计算图中 y 的值
print("y的值:", y.item())
# 反向传播计算梯度
y.backward()
# 输出x的梯度
print("x的梯度:", x.grad)
