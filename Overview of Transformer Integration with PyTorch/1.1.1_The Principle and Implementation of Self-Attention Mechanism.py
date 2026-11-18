import math
import torch
import torch.nn.functional as F

# 1. 模拟输入序列（4个词，维度为3）
embedding = torch.tensor(
    [
        [1.0, 0.5, 0.2],
        [0.3, 1.0, 0.5],
        [0.2, 0.3, 1.0],
        [0.5, 0.7, 0.3],
    ],
    requires_grad=True,
)

# 2. 定义 Q, K, V 的线性变换矩阵
W_q = torch.nn.Linear(3, 3, bias=False)
W_k = torch.nn.Linear(3, 3, bias=False)
W_v = torch.nn.Linear(3, 3, bias=False)

# 3. 线性投影生成 Q, K, V
Q = W_q(embedding)
K = W_k(embedding)
V = W_v(embedding)
print(Q.size)
# 4. 计算缩放点积注意力
d_k = Q.size(-1)
# 使用 math.sqrt 计算缩放因子，代码更加干净
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

# 5. Softmax 归一化
attention_weights = F.softmax(attention_scores, dim=-1)

# 6. 加权求和输出
attention_output = torch.matmul(attention_weights, V)

print("查询矩阵 Q:\n", Q)
print("键矩阵 K:\n", K)
print("值矩阵 V:\n", V)
print("注意力得分:\n", attention_scores)
print("归一化后的注意力权重:\n", attention_weights)
print("自注意力输出:\n", attention_output)