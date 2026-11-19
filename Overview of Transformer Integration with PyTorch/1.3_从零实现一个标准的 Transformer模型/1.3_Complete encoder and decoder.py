import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1. 通用多头注意力机制 (同时支持 Self-Attention 和 Cross-Attention)
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
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


# 2. 前馈神经网络
class FeedForward(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# 3. Transformer 编码器层与编码器
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, mask=mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):

    def __init__(
        self, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


# 4. Transformer 解码器层
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
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
        # 1. 掩码自注意力 (Masked Self-Attention): Q, K, V 均来自 tgt
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. 交叉注意力 (Cross-Attention): Q 来自 tgt，K, V 来自 memory
        tgt2 = self.encoder_attn(
            query_x=tgt, key_x=memory, value_x=memory, mask=memory_mask
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. 前馈神经网络
        tgt2 = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt


# 5. 完整的 Transformer Seq2Seq 模型
class Transformer(nn.Module):

    def __init__(
        self, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, tgt, tgt_mask=None, src_mask=None):
        memory = self.encoder(src, mask=src_mask)
        for layer in self.decoder:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return tgt


# 掩码生成函数 (生成下三角矩阵)
def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask


# --- 运行测试 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 512
num_heads = 8
num_layers = 6
dim_feedforward = 2048
dropout = 0.1

model = Transformer(
    d_model, num_heads, num_layers, dim_feedforward, dropout
).to(device)

src = torch.rand(32, 10, d_model).to(device)
tgt = torch.rand(32, 10, d_model).to(device)
tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(src, tgt, tgt_mask=tgt_mask)
    loss = criterion(output, tgt)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
