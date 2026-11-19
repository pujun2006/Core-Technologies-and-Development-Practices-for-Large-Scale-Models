import math
import matplotlib
matplotlib.use("TkAgg")  # 强制使用标准的 Tkinter 窗口渲染图形
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置 matplotlib 支持中文显示（防止标题乱码）
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 定义标准的多头注意力类
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        # 定义查询、键、值的线性映射
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 补充标准的最终多头融合输出层
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # 1. 线性映射并重塑为多头格式 [batch_size, num_heads, seq_len, d_head]
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

        # 2. 生成缩放点积注意力矩阵 [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = F.softmax(scores, dim=-1)

        # 3. 计算加权值
        attention_output = torch.matmul(attention_weights, V)

        # 4. 拼接多头并线性变换
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.out_proj(attention_output)

        return attention_weights, output


# 可视化注意力矩阵
def visualize_attention_matrix(attention_matrix):
    num_heads = attention_matrix.size(1)
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4), sharey=True)
    fig.suptitle("多头注意力矩阵可视化", fontsize=14, fontweight="bold")

    # 针对每个头绘制注意力权重矩阵
    for i in range(num_heads):
        # 提取第 i 个头的矩阵 (seq_len, seq_len)
        matrix = attention_matrix[0, i].detach().cpu().numpy()

        ax = axes[i] if num_heads > 1 else axes
        sns.heatmap(
            matrix,
            cmap="viridis",
            cbar=True,
            ax=ax,
            square=True,  # 保持单元格为正方形
            annot=True,  # 显示具体概率数值
            fmt=".2f",  # 保留两位小数
            vmin=0.0,
            vmax=1.0,  # 规范色标范围
        )
        ax.set_title(f"Head {i+1}")
        ax.set_xlabel("Key 位置")
        if i == 0:
            ax.set_ylabel("Query 位置")

    plt.tight_layout()
    plt.show()


# 测试函数
def test_attention_visualization():
    batch_size = 1
    seq_len = 6
    d_model = 64
    num_heads = 4

    # 初始化多头注意力模块
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    # 模拟输入数据
    x = torch.rand(batch_size, seq_len, d_model)

    # 计算注意力矩阵
    attention_weights, attention_output = multi_head_attention(x)
    print("注意力权重矩阵形状:", attention_weights.shape)  # [1, 4, 6, 6]
    print("最终输出矩阵形状:", attention_output.shape)  # [1, 6, 64]

    # 可视化注意力矩阵
    visualize_attention_matrix(attention_weights)


# 运行测试
test_attention_visualization()