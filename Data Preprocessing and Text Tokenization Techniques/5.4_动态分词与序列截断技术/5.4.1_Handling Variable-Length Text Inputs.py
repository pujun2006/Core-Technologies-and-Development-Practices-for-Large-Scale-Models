from transformers import AutoTokenizer

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "自然语言处理是一门研究计算机如何理解和生成人类语言的学科。",
    "变长文本输入在批量处理时需要特殊处理。",
    "Hugging Face的分词器可以灵活处理不同长度的文本序列，提高数据利用率。",
]


def tokenize_texts_dynamic(texts, tokenizer, max_length=512):
    """对输入文本进行动态分词和填充，设置安全截断上限"""
    inputs = tokenizer(
        texts,
        padding="longest",  # 按当前 Batch 内最长句子填充（最节省显存）
        truncation=True,  # 开启截断
        max_length=max_length,  # 设置安全上限，防止超长文本导致 OOM
        return_tensors="pt",  # 返回 PyTorch Tensor
    )
    return inputs


# 调用动态分词
tokenized_inputs = tokenize_texts_dynamic(texts, tokenizer)

# 打印对齐后的张量形状 (Batch_Size, Max_Seq_Len_In_Batch)
print("动态对齐后的 input_ids 形状:", tokenized_inputs["input_ids"].shape)

# 逐条验证 Padding 效果
for i, text in enumerate(texts):
    print(f"\n原始文本: {text}")
    print(f"Token 数量（含 CLS/SEP/PAD）: {len(tokenized_inputs['input_ids'][i])}")
    print(f"input_ids: {tokenized_inputs['input_ids'][i]}")
    print(f"attention_mask: {tokenized_inputs['attention_mask'][i]}")