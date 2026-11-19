import math
import torch
import torch.nn as nn
import torch.optim as optim


# 1. 通用多头注意力模块 (支持 Self-Attention 和 Cross-Attention)
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
        if key_x is None:
            key_x = query_x
        if value_x is None:
            value_x = query_x

        batch_size, seq_len_q, d_model = query_x.size()
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

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 修复了 ./ transpose 的语法错误
        attention_output = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, d_model)
        )
        return self.out_proj(attention_output)


# 2. 前馈网络与残差连接
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


# 3. 补全 EncoderLayer 和 DecoderLayer
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
        tgt = self.norm1(tgt, self.self_attn(tgt, mask=tgt_mask))
        tgt = self.norm2(
            tgt,
            self.encoder_attn(
                query_x=tgt, key_x=memory, value_x=memory, mask=memory_mask
            ),
        )
        tgt = self.norm3(tgt, self.feed_forward(tgt))
        return tgt


# 4. 编码器与解码器容器
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


# 5. 因果掩码生成函数
def generate_square_subsequent_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)


# 6. 完整的 Transformer 模型 (映射输出至 vocab_size)
class Transformer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        vocab_size,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        # 将隐藏状态特征投影到词表分类维度
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask=src_mask)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        return self.fc_out(output)


# --- 训练逻辑演示 ---
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    vocab_size = 10000  # 假设词表大小为 10000

    model = Transformer(
        d_model, num_heads, num_layers, dim_feedforward, vocab_size, dropout
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 模拟输入特征 (Batch=16, SeqLen=10, d_model=512)
    src = torch.rand(16, 10, d_model)
    tgt = torch.rand(16, 10, d_model)

    # 模拟真正的分类目标 Token 标签 (Batch=16, SeqLen=10)，值为 0~vocab_size-1 之间的整数
    target_labels = torch.randint(0, vocab_size, (16, 10))

    tgt_mask = generate_square_subsequent_mask(10)

    # 前向传播
    optimizer.zero_grad()
    # output 形状: (16, 10, vocab_size)
    output = model(src, tgt, tgt_mask=tgt_mask)

    # 展平计算交叉熵损失
    # output: (16*10, vocab_size), target_labels: (16*10,)
    loss = criterion(output.view(-1, vocab_size), target_labels.view(-1))

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    print("训练损失:", loss.item())
