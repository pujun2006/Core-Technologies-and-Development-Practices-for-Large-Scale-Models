import math
import torch
import torch.nn as nn


# 多头自注意力实现
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        # 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, "d_model必须是num_heads的整数倍"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # 定义查询、键和值的线性变换层
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 最终输出层
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # 生成查询、键和值的映射，并按照头数分割
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

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 使用注意力权重加权值 (修复了这里的续行符语法错误)
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_model)
        )

        # 通过最终线性层投影输出
        return self.out_proj(attention_output)


# --- 测试多头自注意力模块 ---
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    multi_head_attn = MultiHeadSelfAttention(d_model, num_heads)

    # 模拟输入张量：批次大小 16，序列长度 20，嵌入维度 512
    input_data = torch.rand(16, 20, d_model)

    # 前向传播
    output = multi_head_attn(input_data)

    # 打印输出结果
    print("多头注意力输出形状:", output.shape)  # torch.Size([16, 20, 512])
    print("多头注意力输出示例:", output[0, 0, :10])