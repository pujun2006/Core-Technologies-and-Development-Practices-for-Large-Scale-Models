import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载预训练 Tokenizer 与 模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# 测试文本（包含未登录专有名词）
test_text = "Huggingface provides state-of-the-art NLP tools."

# ----------------------------------------------------
# 验证 1：扩词表前的状态（对比基准）
# ----------------------------------------------------
print("=== 扩词表前 ===")
print("1. 原始词表大小:", len(tokenizer))
# 'huggingface' 被拆分为多个子词 ['hug', '##ging', '##face']
print("2. 原始分词结果:", tokenizer.tokenize(test_text))
print(
    "3. 原始 'huggingface' ID:",
    tokenizer.convert_tokens_to_ids("huggingface"),
)  # 返回 100 (即 [UNK])


# ----------------------------------------------------
# 2. 执行扩词表操作
# ----------------------------------------------------
custom_vocab = ["huggingface", "tokenization"]
num_added = tokenizer.add_tokens(custom_vocab)
print(f"\n成功向词表添加了 {num_added} 个新词！\n")


# ----------------------------------------------------
# 验证 2：扩词表后的 3 维验证（直观看出扩词表效果）
# ----------------------------------------------------
print("=== 扩词表后验证 ===")
# 维度 1：词表总长度增加
print("1. 更新后词表大小:", len(tokenizer))  # 30522 + 2 = 30524

# 维度 2：Token 不再被拆碎，整体切分
print("2. 更新后分词结果:", tokenizer.tokenize(test_text))  # 包含了完整的 'huggingface'

# 维度 3：新词获得了专属的新 ID（位于原始词表末尾）
new_id = tokenizer.convert_tokens_to_ids("huggingface")
print("3. 新词 'huggingface' 分配的 Token ID:", new_id)
print("   反向解码 ID 验证:", tokenizer.decode([new_id]))


# ----------------------------------------------------
# 3. 最关键工程步骤：同步更新模型的 Embedding 矩阵
# ----------------------------------------------------
print("\n=== 模型 Embedding 同步调整 ===")
print("调整前模型 Embedding 形状:", model.get_input_embeddings().weight.shape)

# ⚠️ 如果不执行这一步，传入新 ID (如 30522) 时模型前向传播会报 IndexError 越界错误
model.resize_token_embeddings(len(tokenizer))

print("调整后模型 Embedding 形状:", model.get_input_embeddings().weight.shape)


# ----------------------------------------------------
# 4. 数据编码与前向传播测试
# ----------------------------------------------------
inputs = tokenizer(
    test_text,
    padding="max_length",
    truncation=True,
    max_length=16,
    return_tensors="pt",
)

# 模型正常接受更新后的输入并进行前向传播
outputs = model(**inputs)
print("\n模型前向传播成功，输出 Logits 形状:", outputs.logits.shape)