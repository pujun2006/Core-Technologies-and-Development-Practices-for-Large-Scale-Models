import torch
import torch.nn.functional as F
import math


# 定义点积注意力机制
def scaled_dot_product_attention(query, key, value, mask=None):
    # 计算查询和键的点积
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用缩放机制
    print("原始点积得分:", scores)

    # 可选掩码应用于得分矩阵，避免关注无关信息
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 通过Softmax函数将得分归一化为权重
    attention_weights = F.softmax(scores, dim=-1)
    print("归一化后的注意力权重:", attention_weights)

    # 将权重应用于值，计算加权和
    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights


# 测试函数
def test_scaled_dot_product_attention():
    # 模拟输入的查询、键和值，假设批次大小为1，序列长度为5，特征维度为64
    batch_size = 1
    seq_len = 5
    d_k = 64

    # 随机生成查询、键和值
    query = torch.rand(batch_size, seq_len, d_k)
    key = torch.rand(batch_size, seq_len, d_k)
    value = torch.rand(batch_size, seq_len, d_k)

    # 调用点积注意力机制并输出结果
    attention_output, attention_weights = scaled_dot_product_attention(
        query, key, value)

    # 打印输出结果
    print("注意力输出:", attention_output)
    print("注意力权重:", attention_weights)


# 运行测试
test_scaled_dot_product_attention()