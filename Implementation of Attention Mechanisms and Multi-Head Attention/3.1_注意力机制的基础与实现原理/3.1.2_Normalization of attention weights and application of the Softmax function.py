import math
import torch
import torch.nn.functional as F


# 定义带Softmax归一化的注意力计算函数
def attention_with_softmax(query, key, value, mask=None):
    # 获取特征维度
    d_k = query.size(-1)

    # 计算点积得分并进行缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print("原始点积得分矩阵:\n", scores)

    # 应用掩码来屏蔽无效信息
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 通过Softmax函数归一化点积得分
    attention_weights = F.softmax(scores, dim=-1)
    print("归一化后的注意力权重矩阵:\n", attention_weights)

    # 计算加权和，生成最终的注意力输出
    attention_output = torch.matmul(attention_weights, value)

    # 修复了缩进错误 ✅
    return attention_output, attention_weights


# 测试函数
def test_attention_with_softmax():
    # 定义查询、键和值张量，批次大小为1，序列长度为4，特征维度为16
    batch_size = 1
    seq_len = 4
    d_k = 16

    # 随机生成输入数据
    query = torch.rand(batch_size, seq_len, d_k)
    key = torch.rand(batch_size, seq_len, d_k)
    value = torch.rand(batch_size, seq_len, d_k)

    # 调用注意力计算函数并打印输出结果
    attention_output, attention_weights = attention_with_softmax(
        query, key, value
    )
    print("最终的注意力输出:\n", attention_output)
    print("归一化后的注意力权重:\n", attention_weights)


# 运行测试
test_attention_with_softmax()
