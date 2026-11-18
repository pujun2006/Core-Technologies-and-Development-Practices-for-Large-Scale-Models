import math
import torch
import torch.nn as nn


# 1. 通用多头注意力模块 (同时支持 Self-Attention 和 Cross-Attention)
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query_x, key_x=None, value_x=None, mask=None):
        # 若未传 key_x 和 value_x，则默认为 Self-Attention
        if key_x is None:
            key_x = query_x
        if value_x is None:
            value_x = query_x

        batch_size, seq_len_q, d_model = query_x.size()
        seq_len_k = key_x.size(1)

        # 生成 Q, K, V 映射
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

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 修复续行符语法错误
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, d_model)
        )

        print("Attention Weights Shape:", attention_weights.shape)
        return self.out_proj(attention_output)


# 2. 前馈网络
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


# 3. 残差连接与层归一化
class ResidualNormLayer(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super(ResidualNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        x = x + self.dropout(sublayer_output)
        return self.layer_norm(x)


# 4. 编码器层
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.residual_norm1 = ResidualNormLayer(d_model, dropout)
        self.residual_norm2 = ResidualNormLayer(d_model, dropout)

    def forward(self, src, src_mask=None):
        print("\n--- Encoder Layer ---")
        src2 = self.self_attn(src, mask=src_mask)
        src = self.residual_norm1(src, src2)
        src2 = self.feed_forward(src)
        src = self.residual_norm2(src, src2)
        return src


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
        print("\n--- Decoder Layer ---")
        # Self-Attention
        tgt2 = self.self_attn(tgt, mask=tgt_mask)
        tgt = self.residual_norm1(tgt, tgt2)

        # Cross-Attention (Q来自于tgt, K和V来自于memory)
        tgt2 = self.encoder_attn(
            query_x=tgt, key_x=memory, value_x=memory, mask=memory_mask
        )
        tgt = self.residual_norm2(tgt, tgt2)

        # Feed-Forward
        tgt2 = self.feed_forward(tgt)
        tgt = self.residual_norm3(tgt, tgt2)
        return tgt


# 6. Transformer 模型
class Transformer(nn.Module):

    def __init__(
        self, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        print("==== Encoding Process ====")
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask=src_mask)

        print("\n==== Decoding Process ====")
        output = tgt
        for layer in self.decoder_layers:
            output = layer(
                output, memory, tgt_mask=tgt_mask, memory_mask=src_mask
            )

        return output


# --- 测试示例数据 ---
if __name__ == "__main__":
    d_model = 64
    num_heads = 4
    num_layers = 2
    dim_feedforward = 128
    dropout = 0.1

    transformer = Transformer(
        d_model, num_heads, num_layers, dim_feedforward, dropout
    )

    src = torch.rand(1, 5, d_model)  # 输入序列
    tgt = torch.rand(1, 5, d_model)  # 目标序列
    src_mask = torch.ones(1, 1, 5, 5)  # 编码器掩码
    tgt_mask = torch.tril(torch.ones(1, 1, 5, 5))  # 解码器因果掩码

    output = transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print("\nFinal Transformer Output Shape:", output.shape)