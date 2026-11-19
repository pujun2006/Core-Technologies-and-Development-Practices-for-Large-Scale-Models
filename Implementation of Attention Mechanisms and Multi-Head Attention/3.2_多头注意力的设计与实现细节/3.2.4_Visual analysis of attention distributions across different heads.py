import math
import matplotlib

# 1. 解决 PyCharm 后端冲突报错（必须在 import pyplot 之前）
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2. 解决 Matplotlib 中文显示乱码问题
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 定义多头注意力模块
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        # 定义查询、键和值的线性映射
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # 查询、键和值的线性映射
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

        # 计算缩放点积注意力矩阵
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = F.softmax(scores, dim=-1)

        # 返回注意力矩阵
        return attention_weights


# 可视化不同头的注意力分布
def visualize_attention_heads(attention_matrix):
    num_heads = attention_matrix.size(1)
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4.5), sharey=True)
    fig.suptitle("不同头的注意力分布可视化", fontsize=14, fontweight="bold")

    for i in range(num_heads):
        ax = axes[i] if num_heads > 1 else axes
        sns.heatmap(
            attention_matrix[0, i].detach().cpu().numpy(),
            cmap="coolwarm",
            cbar=True,
            ax=ax,
            square=True,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"Head {i+1}")
        ax.set_xlabel("Key位置")
        if i == 0:
            ax.set_ylabel("Query位置")

    plt.tight_layout()
    plt.show()


# 测试函数，生成并可视化不同头的注意力分布
def test_attention_heads_visualization():
    batch_size = 1
    seq_len = 8
    d_model = 64
    num_heads = 4

    # 初始化多头注意力模块
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    # 模拟输入数据
    x = torch.rand(batch_size, seq_len, d_model)

    # 计算注意力矩阵
    attention_weights = multi_head_attention(x)
    print("不同头的注意力权重矩阵形状:", attention_weights.shape)  # [1, 4, 8, 8]

    # 可视化不同头的注意力分布
    visualize_attention_heads(attention_weights)


# 运行测试
if __name__ == "__main__":
    test_attention_heads_visualization()
