import math
import torch
import torch.nn as nn


# 1. 多头自注意力模块
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须是num_heads的整数倍"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
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

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 修复续行符语法错误
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_model)
        )

        return self.out_proj(attention_output)


# 2. 残差连接和层归一化模块 (Post-LN 结构)
class ResidualNormLayer(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super(ResidualNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # 残差连接: 将输入x与子层的输出相加
        x = x + self.dropout(sublayer_output)
        # 层归一化: 保持特征分布的稳定性
        return self.layer_norm(x)


# 3. 前馈网络模块
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


# 4. Transformer编码器层，集成残差连接与层归一化
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        # 残差连接与层归一化模块
        self.residual_norm1 = ResidualNormLayer(d_model, dropout)
        self.residual_norm2 = ResidualNormLayer(d_model, dropout)

    def forward(self, src):
        # 多头注意力模块 + 残差连接和层归一化
        src2 = self.self_attn(src)
        src = self.residual_norm1(src, src2)

        # 前馈网络模块 + 残差连接和层归一化
        src2 = self.feed_forward(src)
        src = self.residual_norm2(src, src2)

        return src


# --- 测试编码器层模块 ---
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    dim_feedforward = 2048
    dropout = 0.1

    # 初始化编码器层
    encoder_layer = TransformerEncoderLayer(
        d_model, num_heads, dim_feedforward, dropout
    )

    # 输入张量: 批次大小16，序列长度10，嵌入维度512
    input_data = torch.rand(16, 10, d_model)

    # 前向传播
    output = encoder_layer(input_data)

    # 输出结果
    print("编码器层输出形状:", output.shape)  # torch.Size([16, 10, 512])
    print("编码器层输出示例:", output[0, 0, :10])
