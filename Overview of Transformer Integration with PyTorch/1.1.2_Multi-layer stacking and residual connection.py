import torch
import torch.nn as nn
import torch.nn.functional as F


# 手动实现的 Transformer 编码器层 (支持 batch_first)
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # 设置 batch_first=True，更符合现代开发习惯
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 前馈神经网络层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 残差连接与归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # 1. 多头自注意力计算
        attn_output, _ = self.multihead_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_output))  # 残差 + Norm

        # 2. 前馈神经网络计算
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_output))  # 残差 + Norm

        return src


# 测试参数
d_model = 4  # 词嵌入维度
num_heads = 2  # 头数 (必须能被 d_model 整除，4 % 2 == 0)
dim_feedforward = 8  # 隐藏层维度
dropout = 0.1

encoder_layer = TransformerEncoderLayer(
    d_model, num_heads, dim_feedforward, dropout
)

# 模拟输入张量：(batch_size=1, seq_len=4, d_model=4)
src = torch.rand((1, 4, d_model))

# 前向传播
output = encoder_layer(src)

print("输入张量 (Shape: {}):\n".format(src.shape), src)
print("\nTransformer编码器层输出 (Shape: {}):\n".format(output.shape), output)