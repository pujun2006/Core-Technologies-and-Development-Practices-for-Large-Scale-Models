from transformers import AutoTokenizer

# 1. 定义模型名称并加载分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 示例文本
texts = [
    "自然语言处理的目的是使计算机能够理解和生成自然语言。",
    "在实际应用中，文本数据通常具有不确定的长度。",
    "通过使用填充和截断技术，确保每个批次内的序列长度一致，提升模型的处理效率。",
]


# 3. 分词与截断填充函数
def tokenize_and_pad(texts, tokenizer, max_length=64):
    """对长文本序列进行分词，并通过填充和截断使序列长度一致"""
    inputs = tokenizer(
        texts,
        padding="max_length",  # 填充至固定长度
        truncation=True,  # 截断长于 max_length 的序列
        max_length=max_length,  # 设置更加合理的序列长度上限
        return_tensors="pt",  # 返回 PyTorch 张量格式
    )
    return inputs


# 4. 调用分词（此处调整 max_length 为更合理的 64，填写10可以查看截断情况）
max_length = 10  # 64
tokenized_inputs = tokenize_and_pad(texts, tokenizer, max_length)

# 5. 逐条输出结果并还原解码文本
print("=== 填充与截断验证 ===\n")
for i, text in enumerate(texts):
    input_ids = tokenized_inputs["input_ids"][i]
    attention_mask = tokenized_inputs["attention_mask"][i]

    # 将 Tensor 解码回人类可读的字符串（保留特殊 Token 以便观察）
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    print(f"原始文本: {text}")
    print(f"张量形状: {input_ids.shape}")
    print(f"还原文本: {decoded_text}")
    print(
        f"有效长度 (Sum of Attention Mask): {attention_mask.sum().item()}"
    )
    print("-" * 60)