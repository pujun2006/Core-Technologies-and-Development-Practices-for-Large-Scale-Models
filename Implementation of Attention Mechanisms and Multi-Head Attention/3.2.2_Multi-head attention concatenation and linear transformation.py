import math
import torch
import torch.nn as nn


# 定义多头注意力模块
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # 增加整除检查断言
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

        # 定义输出的线性变换
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 1. 线性映射查询、键和值
        Q = self.query(x)  # [batch_size, seq_len, d_model]
        K = self.key(x)
        V = self.value(x)

        # 2. 重塑查询、键和值为多头格式
        Q = Q.view(
            batch_size, seq_len, self.num_heads, self.d_head
        ).transpose(1, 2)
        K = K.view(
            batch_size, seq_len, self.num_heads, self.d_head
        ).transpose(1, 2)
        V = V.view(
            batch_size, seq_len, self.num_heads, self.d_head
        ).transpose(1, 2)

        # 3. 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 支持 Masking 操作
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # 4. 将多头输出拼接为完整维度 (修复了正斜杠语法错误 ✅)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_model)
        )
        print("多头拼接后的输出形状:", attention_output.shape)

        # 5. 应用线性变换统一输出维度
        output = self.out_proj(attention_output)
        print("线性变换后的最终输出形状:", output.shape)

        return output


# 测试函数
def test_multi_head_attention():
    batch_size = 1
    seq_len = 5
    d_model = 64
    num_heads = 8

    # 初始化多头注意力模块
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    # 模拟输入数据
    x = torch.rand(batch_size, seq_len, d_model)

    # 前向传播计算
    output = multi_head_attention(x)
    print("多头注意力模块最终输出形状:", output.shape)


# 运行测试
test_multi_head_attention()