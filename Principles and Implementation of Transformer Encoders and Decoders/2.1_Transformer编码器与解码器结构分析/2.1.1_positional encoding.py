import math
import torch
import torch.nn as nn


# 合并后的位置编码模块（支持 batch_first=True）
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1. 计算位置编码矩阵，初始形状: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数索引用 sin，奇数索引用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 2. 增加 batch 维度，形状调整为: (1, max_len, d_model)
        # 适配 batch_first=True 的标准输入形状 (batch_size, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # 3. 注册为 buffer，不更新梯度但跟随模型 save/load 保存
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x 的形状: (batch_size, seq_len, d_model)"""
        # 拦截匹配当前 seq_len 的位置编码并相加
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --- 测试与验证 ---
if __name__ == "__main__":
    d_model = 512
    seq_len = 20
    batch_size = 32

    # 实例化位置编码模块
    pos_encoding = PositionalEncoding(d_model=d_model, dropout=0.1)

    # 模拟输入张量: (batch_size=32, seq_len=20, d_model=512)
    input_data = torch.zeros(batch_size, seq_len, d_model)

    # 前向传播
    output = pos_encoding(input_data)

    # 验证维度输出
    print("输入张量形状:", input_data.shape)  # torch.Size([32, 20, 512])
    print("输出张量形状:", output.shape)  # torch.Size([32, 20, 512])
    print(
        "\n第一个样本前两个位置的部分编码特征片段:\n", output[0, :2, :4]
    )
