import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1. 修正后的位置编码 (适应 batch_first=True)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# 2. 通用多头注意力模块 (同时支持 Self-Attention 和 Cross-Attention)
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query_x, key_x, value_x, mask=None):
        batch_size, seq_len_q, _ = query_x.size()
        seq_len_k = key_x.size(1)

        # 线性变换并拆分多头
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

        # 计算 Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 施加掩码 (Masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, -1)
        )

        return self.out(attention_output)


# 3. 前馈神经网络
class FeedForward(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# 4. 编码器层
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Self-Attention: Q, K, V 均来自 src
        src2 = self.self_attn(src, src, src, mask=mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


# 5. 解码器层
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. Masked Self-Attention
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross-Attention: Q 来自 tgt，K 和 V 来自 encoder 的 memory
        tgt2 = self.encoder_attn(
            query_x=tgt, key_x=memory, value_x=memory, mask=memory_mask
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. Feed Forward
        tgt2 = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


# 6. 完整 Transformer 模型
class Transformer(nn.Module):

    def __init__(
        self, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

    def forward(self, src, tgt, tgt_mask=None, src_mask=None):
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = src
        for layer in self.encoder:
            memory = layer(memory, mask=src_mask)

        output = tgt
        for layer in self.decoder:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return output


# 下三角因果掩码生成函数
def generate_square_subsequent_mask(sz):
    mask = torch.tril(torch.ones(sz, sz)).unsqueeze(0).unsqueeze(0)
    return mask


# --- 测试 Transformer 模型 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    model = Transformer(
        d_model, num_heads, num_layers, dim_feedforward, dropout
    ).to(device)

    # 模拟输入：(batch_size=32, seq_len=20, d_model=512)
    batch_size = 32
    seq_len = 20
    src = torch.rand((batch_size, seq_len, d_model)).to(device)
    tgt = torch.rand((batch_size, seq_len, d_model)).to(device)

    # 生成解码器 Self-Attention 用的下三角掩码
    tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

    # 前向传播测试
    output = model(src, tgt, tgt_mask=tgt_mask)

    print("Transformer模型输出形状:", output.shape)  # torch.Size([32, 20, 512])
    print("Transformer模型输出示例:", output[0, 0, :10])