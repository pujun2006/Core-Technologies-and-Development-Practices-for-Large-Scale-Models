import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

# 1. 初始化模型配置，显式开启 output_attentions
config = BertConfig.from_pretrained(
    "bert-base-uncased", output_attentions=True
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", config=config)
model.eval()

# 2. 准备输入句子
text = "Deep learning models have transformed NLP applications drastically."
inputs = tokenizer(text, return_tensors="pt")


# 3. 提取注意力权重的函数
def extract_attention_weights(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention_weights = outputs.attentions

    # 防御性检查：确保成功提取到了注意力权重
    if attention_weights is None or len(attention_weights) == 0:
        raise ValueError(
            "未能提取到注意力权重，请检查模型配置是否开启了 output_attentions=True。"
        )

    return attention_weights


# 4. 提取权重
attention_weights = extract_attention_weights(model, inputs)

# 5. 检查维度与输出
num_layers = len(attention_weights)
num_heads = attention_weights[0].shape[1]
seq_len = attention_weights[0].shape[2]

print(
    f"模型包含 {num_layers} 层，每层有 {num_heads} 个注意力头，每个头的序列长度为 {seq_len}"
)

# 6. 安全展示（只打印第 1 层第 1 个头的形状与部分数值，避免控制台刷屏卡死）
print("\n第 1 层，第 1 个注意力头矩阵形状:", attention_weights[0][0, 0].shape)
print("矩阵前 3x3 部分数值:\n", attention_weights[0][0, 0][:3, :3].cpu().numpy())