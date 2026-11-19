import torch
from transformers import AutoModel, AutoTokenizer

# 1. 加载 Tokenizer 与预训练模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)  # 加载真实的 BERT 预训练模型

# 2. 获取预训练好的真实 Embedding 层（保留海量语义信息）
pretrained_embedding_layer = model.get_input_embeddings()

# 3. 示例文本与分词 (调整合理的 max_length)
texts = ["自然语言处理是一门研究计算机理解和生成人类语言的学科。"]
tokenized_inputs = tokenizer(
    texts, padding="max_length", truncation=True, max_length=20, return_tensors="pt"
)

input_ids = tokenized_inputs["input_ids"]

# 4. 查表获取真实预训练词嵌入
with torch.no_grad():
    embeddings = pretrained_embedding_layer(input_ids)

# 5. 打印输出
print("输入 input_ids 形状:", input_ids.shape)  # torch.Size([1, 20])
print("词嵌入 output 形状:", embeddings.shape)    # torch.Size([1, 20, 768])
print("\n前 5 个 Token 的预训练 Embedding (部分维度):\n", embeddings[0, :5, :4])