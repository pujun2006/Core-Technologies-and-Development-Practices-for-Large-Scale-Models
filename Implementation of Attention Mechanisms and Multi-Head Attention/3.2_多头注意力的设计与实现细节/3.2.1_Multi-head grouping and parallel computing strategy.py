import math
import torch
import torch.nn as nn


# 定义多头注意力模块
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # 增加整除检查
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        # 设置每个头的维度
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        # 定义查询、键和值的线性映射
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query_x, key_x=None, value_x=None, mask=None):
        # 如果未传入 key_x 和 value_x，则默认为 Self-Attention
        if key_x is None:
            key_x = query_x
        if value_x is None:
            value_x = query_x

        batch_size, seq_len_q, d_model = query_x.size()
        seq_len_k = key_x.size(1)

        # 1. 线性映射查询、键和值
        Q = self.query(query_x)  # [batch_size, seq_len_q, d_model]
        K = self.key(key_x)  # [batch_size, seq_len_k, d_model]
        V = self.value(value_x)  # [batch_size, seq_len_k, d_model]

        # 2. 重塑查询、键和值为多头格式 [batch_size, num_heads, seq_len, d_head]
        Q = Q.view(
            batch_size, seq_len_q, self.num_heads, self.d_head
        ).transpose(1, 2)
        K = K.view(
            batch_size, seq_len_k, self.num_heads, self.d_head
        ).transpose(1, 2)
        V = V.view(
            batch_size, seq_len_k, self.num_heads, self.d_head
        ).transpose(1, 2)

        # 3. 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 支持 Masking 操作
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # 4. 输出拼接 (修复了正斜杠语法错误 ✅)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )

        return self.out_proj(attention_output)


# 测试函数
def test_multi_head_attention():
    batch_size = 1
    seq_len = 5
    d_model = 64
    num_heads = 8

    # 初始化多头注意力模块
    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    x = torch.rand(batch_size, seq_len, d_model)  # 模拟输入数据

    # 前向传播计算
    output = multi_head_attention(x)
    print("多头注意力输出形状:", output.shape)
    print("多头注意力输出示例:\n", output)


# 运行测试
test_multi_head_attention()
