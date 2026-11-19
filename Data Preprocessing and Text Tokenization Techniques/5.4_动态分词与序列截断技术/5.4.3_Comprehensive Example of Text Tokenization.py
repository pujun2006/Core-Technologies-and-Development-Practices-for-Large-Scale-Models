import re
import torch
from transformers import AutoModel, AutoTokenizer

# 1. 加载 Tokenizer 和 BERT 预训练模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# 停用词集合
stopwords = {"的", "和", "在", "是", "了", "也", "有", "与", "对", "就"}

texts = [
    "自然语言处理是一门研究计算机如何理解和生成人类语言的学科。",
    "在实际应用中，文本数据通常具有不确定的长度。",
    "通过使用填充和截断技术，确保每个批次内的序列长度一致，提升模型的处理效率。",
]


# 2. 正确的中文文本清洗函数（按字符切分过滤）
def clean_chinese_text(text, stopwords):
    # 移除标点符号
    text = re.sub(r"[^\w\s]", "", text)
    # 中文按单字切分 (若要按词切分，需引入 jieba)
    chars = list(text)
    # 去除停用词与空格
    filtered_chars = [
        char for char in chars if char not in stopwords and not char.isspace()
    ]
    # 重新拼接为字符串（也可以直接传给 tokenizer）
    return "".join(filtered_chars)


# 清洗文本
cleaned_texts = [clean_chinese_text(text, stopwords) for text in texts]


# 3. 分词与填充
def tokenize_and_pad(texts, tokenizer, max_length=20):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


tokenized_inputs = tokenize_and_pad(cleaned_texts, tokenizer, max_length=20)

# 4. 使用 BERT 原生的预训练 Embedding 层（保留语义信息）
pretrained_embedding_layer = bert_model.get_input_embeddings()

with torch.no_grad():
    embeddings = pretrained_embedding_layer(tokenized_inputs["input_ids"])

# 5. 输出验证
print("=== 清洗对比 ===")
print("原始文本 1:", texts[0])
print("清洗文本 1:", cleaned_texts[0])  # 成功去除了 '是'、'的' 等字

print("\n=== 张量与 Embeddings ===")
print("Input IDs 形状:", tokenized_inputs["input_ids"].shape)
print("Embedding 输出形状:", embeddings.shape)