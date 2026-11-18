import math
import torch
import torch.nn as nn


# 1. 通用多头注意力模块 (同时支持 Self-Attention 和 Cross-Attention)
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须是num_heads的整数倍"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query_x, key_x=None, value_x=None, mask=None):
        # 如果未传入 key 和 value，则默认进行 Self-Attention
        if key_x is None:
            key_x = query_x
        if value_x is None:
            value_x = query_x

        batch_size, seq_len_q, _ = query_x.size()
        seq_len_k = key_x.size(1)

        # 变换 Q, K, V
        Q = (
            self.query(query_x)
            .view(batch_size, seq_len_q, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        K = (
            self.key(key_x)
            .view(batch_size, seq_len_k, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        V = (
            self.value(value_x)
            .view(batch_size, seq_len_k, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 应用掩码 (Masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, -1)
        )

        return self.out_proj(attention_output)


# 2. 前馈网络模块 (FeedForward)
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


# 3. 残差连接和层归一化模块 (ResidualNormLayer)
class ResidualNormLayer(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super(ResidualNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        x = x + self.dropout(sublayer_output)
        return self.layer_norm(x)


# 4. 下三角因果掩码生成函数
def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask


# 5. 解码器层
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.residual_norm1 = ResidualNormLayer(d_model, dropout)
        self.residual_norm2 = ResidualNormLayer(d_model, dropout)
        self.residual_norm3 = ResidualNormLayer(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. Self-Attention (Masked)
        tgt2 = self.self_attn(tgt, mask=tgt_mask)
        tgt = self.residual_norm1(tgt, tgt2)

        # 2. Cross-Attention (Q来自于tgt, K和V来自于memory)
        tgt2 = self.encoder_attn(
            query_x=tgt, key_x=memory, value_x=memory, mask=memory_mask
        )
        tgt = self.residual_norm2(tgt, tgt2)

        # 3. Feed Forward
        tgt2 = self.feed_forward(tgt)
        tgt = self.residual_norm3(tgt, tgt2)

        return tgt


# 6. 完整的 Transformer 解码器
class TransformerDecoder(nn.Module):

    def __init__(
        self, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return tgt


# --- 测试解码器自回归生成过程 ---
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # 初始化解码器和掩码
    decoder = TransformerDecoder(
        d_model, num_heads, num_layers, dim_feedforward, dropout
    )
    tgt_mask = generate_square_subsequent_mask(10)  # 序列长度10

    # 模拟输入数据
    memory = torch.rand(16, 10, d_model)  # 编码器输出
    tgt = torch.rand(16, 10, d_model)  # 解码器输入

    # 前向传播
    output = decoder(tgt, memory, tgt_mask=tgt_mask)

    # 输出结果
    print("解码器输出形状:", output.shape)  # torch.Size([16, 10, 512])
    print("解码器输出示例:", output[0, 0, :10])