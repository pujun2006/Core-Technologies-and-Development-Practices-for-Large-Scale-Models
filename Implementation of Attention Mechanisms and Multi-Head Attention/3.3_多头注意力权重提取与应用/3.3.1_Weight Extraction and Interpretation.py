import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义多头注意力类，包含注意力权重提取与多头融合
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        # 定义查询、键和值的线性映射
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 补充完整的多头融合输出层
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 1. 查询、键和值的线性映射
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        # 2. 计算缩放点积注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 可选：应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        # 3. 计算加权值并拼接输出
        attention_output = torch.matmul(attention_weights, V)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.out_proj(attention_output)

        return attention_weights, output


# 提取并打印注意力权重
def extract_attention_weights(attention_weights):
    num_heads = attention_weights.size(1)

    for i in range(num_heads):
        print(f"\n--- Head {i+1} 的注意力权重矩阵 ---")
        # 使用 torch.set_printoptions 精简打印保留 4 位小数
        torch.set_printoptions(precision=4, sci_mode=False)
        print(attention_weights[0, i].detach())


# 测试函数，提取和解读多头注意力权重
def test_attention_weights_extraction():
    batch_size = 1
    seq_len = 6
    d_model = 64
    num_heads = 4

    # 初始化多头注意力模块
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    # 模拟输入数据
    x = torch.rand(batch_size, seq_len, d_model)

    # 计算注意力权重与最终输出
    attention_weights, output = multi_head_attention(x)
    print("所有头的注意力权重矩阵形状:", attention_weights.shape)  # [1, 4, 6, 6]

    # 提取并逐个打印注意力权重
    extract_attention_weights(attention_weights)


# 运行测试
test_attention_weights_extraction()
