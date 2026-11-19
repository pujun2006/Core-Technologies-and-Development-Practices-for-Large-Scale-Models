import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义多头注意力模块，包含正则化与梯度裁剪的优化
class OptimizedMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1, reg_factor=0.1):
        super(OptimizedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.reg_factor = reg_factor  # 正则化平滑系数 [0, 1)

        # 定义查询、键和值的线性映射
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 输出层的线性变换与丢弃层
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

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

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = F.softmax(scores, dim=-1)

        # 正则化：利用均匀分布平滑注意力（Label Smoothing 思想），避免权重过分集中
        # 修复了二次 Softmax 逻辑与正斜杠语法错误 ✅
        uniform_dist = torch.full_like(
            attention_weights, 1.0 / seq_len
        )  # 均匀分布
        regularized_weights = (
            1 - self.reg_factor
        ) * attention_weights + self.reg_factor * uniform_dist

        # 加权求和计算注意力输出
        attention_output = torch.matmul(regularized_weights, V)

        # 修复了正斜杠语法错误 ✅
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_model)
        )

        # 丢弃处理和输出
        output = self.dropout(self.out_proj(attention_output))
        return output, regularized_weights


# 修正后的梯度裁剪函数
def clip_gradients(model, max_norm=1.0):
    # 修复：直接传入 parameters 迭代器，按全局范数统一裁剪，不需要在外部循环遍历 ✅
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return total_norm


# 测试函数，验证正则化与梯度裁剪效果
def test_attention_optimization():
    batch_size = 1
    seq_len = 6
    d_model = 64
    num_heads = 4
    max_norm = 0.5  # 梯度裁剪的阈值

    # 初始化优化的多头注意力模块
    optimized_attention = OptimizedMultiHeadAttention(
        d_model, num_heads, reg_factor=0.1
    )

    # 模拟输入数据
    x = torch.rand(batch_size, seq_len, d_model, requires_grad=True)

    # 前向传播计算
    output, regularized_weights = optimized_attention(x)
    print("正则化后的注意力权重矩阵形状:", regularized_weights.shape)
    print("多头注意力输出形状:", output.shape)

    # 计算损失并进行反向传播
    loss = output.sum()
    loss.backward()

    # 执行梯度裁剪
    grad_norm = clip_gradients(optimized_attention, max_norm)
    print(f"\n裁剪前的全局梯度范数 (Global Norm): {grad_norm:.4f}")
    print(f"梯度已成功裁剪至 max_norm={max_norm}\n")

    # 打印模型中部分参数的梯度示例
    print("--- 梯度裁剪后参数梯度示例 ---")
    for name, param in optimized_attention.named_parameters():
        if param.grad is not None:
            print(
                f"{name} 的梯度形状: {param.grad.shape}, 范数: {param.grad.norm():.4f}"
            )


# 运行测试
if __name__ == "__main__":
    test_attention_optimization()