import math
import torch
import torch.nn as nn


# =====================================================================
# 1. 通用多头注意力模块 (支持 Self-Attention 与 Cross-Attention)
# =====================================================================
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须是 num_heads 的整数倍"
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

        # 投影并重塑形状为: [batch_size, num_heads, seq_len, d_head]
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

        # 计算 Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 应用掩码 (mask 为 bool 类型，元素为 False 的位置填充为 -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 拼接多头输出
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, d_model)
        )

        return self.out_proj(attention_output)


# =====================================================================
# 2. 前馈网络 (Feed Forward) 与残差归一化层 (ResidualNormLayer)
# =====================================================================
class FeedForward(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class ResidualNormLayer(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super(ResidualNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))


# =====================================================================
# 3. 编码器层 (EncoderLayer) 与解码器层 (DecoderLayer)
# =====================================================================
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = ResidualNormLayer(d_model, dropout)
        self.norm2 = ResidualNormLayer(d_model, dropout)

    def forward(self, src, src_mask=None):
        src = self.norm1(src, self.self_attn(src, mask=src_mask))
        src = self.norm2(src, self.feed_forward(src))
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = ResidualNormLayer(d_model, dropout)
        self.norm2 = ResidualNormLayer(d_model, dropout)
        self.norm3 = ResidualNormLayer(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. Masked Self-Attention
        tgt = self.norm1(tgt, self.self_attn(tgt, mask=tgt_mask))
        # 2. Cross-Attention (Q来自于tgt, K和V来自于memory)
        tgt = self.norm2(
            tgt,
            self.encoder_attn(
                query_x=tgt, key_x=memory, value_x=memory, mask=memory_mask
            ),
        )
        # 3. Feed Forward
        tgt = self.norm3(tgt, self.feed_forward(tgt))
        return tgt


# =====================================================================
# 4. 编码器 (Encoder) 与解码器 (Decoder) 堆叠容器
# =====================================================================
class TransformerEncoder(nn.Module):

    def __init__(
        self, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask)
        return src


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


# =====================================================================
# 5. 掩码生成函数 (保证输出格式均为 torch.bool)
# =====================================================================
# 生成填充掩码 Padding Mask (返回 bool 张量)
def create_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


# 生成下三角自回归掩码 Subsequent Mask (显式声明 dtype=torch.bool)
def generate_square_subsequent_mask(size):
    return (
        torch.tril(torch.ones(size, size, dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
    )


# =====================================================================
# 6. 整合 Embedding 与全连接层的完整 Transformer 模型
# =====================================================================
class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = TransformerEncoder(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 1. 词嵌入映射 [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        # 2. 前向编码与解码
        memory = self.encoder(src_emb, src_mask=src_mask)
        output = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=src_mask
        )

        # 3. 映射到词表分类输出维度
        return self.fc_out(output)


# =====================================================================
# 7. 测试与运行
# =====================================================================
if __name__ == "__main__":
    vocab_size = 100
    d_model = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # 初始化模型
    model = Transformer(
        vocab_size, d_model, num_heads, num_layers, dim_feedforward, dropout
    )

    # 模拟输入序列 (16 个样本，长度为 10，数值为 Token ID)
    src = torch.randint(0, 10, (16, 10))
    tgt = torch.randint(0, 10, (16, 10))

    # 生成 Padding Mask 与 Subsequent Mask (均保证为 bool 类型)
    src_mask = create_padding_mask(src, pad_token=0)
    tgt_pad_mask = create_padding_mask(tgt, pad_token=0)
    tgt_subsequent_mask = generate_square_subsequent_mask(tgt.size(1))

    # 顺利进行按位与逻辑组合 (bool & bool)
    combined_tgt_mask = tgt_pad_mask & tgt_subsequent_mask

    # 执行前向传播
    output = model(src, tgt, src_mask=src_mask, tgt_mask=combined_tgt_mask)

    print("模型运行成功！")
    print("模型输出形状:", output.shape)  # 期望输出: torch.Size([16, 10, 100])
    print("第一个Token的输出预测向量(前10维):", output[0, 0, :10])