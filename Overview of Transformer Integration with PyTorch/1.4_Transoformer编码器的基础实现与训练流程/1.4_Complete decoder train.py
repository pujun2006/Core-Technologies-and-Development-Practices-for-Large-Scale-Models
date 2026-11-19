import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1. 定义支持 Mask 的多头注意力机制
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_head = d_model // num_heads  # 每个注意力头的维度
        self.num_heads = num_heads

        # 查询、键和值的线性变换层
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 输出层
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 线性变换 + 分头 (batch_size, num_heads, seq_len, d_head)
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

        # 计算缩放点积注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 加上 Mask 支持 (可选)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和并拼接多头
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
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


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

    def forward(self, src, mask=None):
        # 多头注意力 + 残差连接 + 层归一化
        src2 = self.self_attn(src, mask)
        src = self.norm1(src + self.dropout1(src2))

        # 前馈神经网络 + 残差连接 + 层归一化
        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(src2))

        return src


# 4. 定义编码器
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

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


# --- 测试与训练循环 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 512
num_heads = 8
num_layers = 6
dim_feedforward = 2048
dropout = 0.1

encoder = TransformerEncoder(
    d_model, num_heads, num_layers, dim_feedforward, dropout
).to(device)

input_data = torch.rand(32, 10, d_model).to(device)
target_data = torch.rand(32, 10, d_model).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters(), lr=0.0001)

encoder.train()
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = encoder(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
