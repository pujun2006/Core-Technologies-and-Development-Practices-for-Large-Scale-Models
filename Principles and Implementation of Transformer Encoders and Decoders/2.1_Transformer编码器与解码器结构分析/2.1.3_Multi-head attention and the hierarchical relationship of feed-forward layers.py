import math
import torch
import torch.nn as nn


# 1. 定义多头自注意力机制
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须是 num_heads 的整数倍"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # 线性变换并转置为 (batch_size, num_heads, seq_len, d_head)
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

        # 修复续行符问题：移除多余的正斜杠 /
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_model)
        )

        return self.out(attention_output)


# 2. 定义前馈神经网络
class FeedForward(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


# 3. 定义 Transformer 编码器层
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # 多头注意力 + 残差连接 + 层归一化
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络 + 残差连接 + 层归一化
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


# --- 测试编码器层 ---
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    dim_feedforward = 2048
    dropout = 0.1

    # 初始化编码器层
    encoder_layer = TransformerEncoderLayer(
        d_model, num_heads, dim_feedforward, dropout
    )

    # 输入张量：批次大小 32，序列长度 20，嵌入维度 512
    input_data = torch.rand(32, 20, d_model)

    # 前向传播
    output = encoder_layer(input_data)

    # 输出结果验证
    print("编码器层输出形状:", output.shape)  # torch.Size([32, 20, 512])
    print("编码器层输出示例:", output[0, 0, :10])
