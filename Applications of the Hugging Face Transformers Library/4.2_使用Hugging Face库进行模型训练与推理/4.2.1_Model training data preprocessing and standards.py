import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

import torch
from transformers import AutoTokenizer

# 定义模型名称
model_name = "bert-base-uncased"

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 示例数据集
texts = [
    "Hugging Face's transformers library simplifies NLP tasks.",
    "Preprocessing is essential for effective model training.",
    "Tokenization helps in converting text to tensor format."
]
labels = [1, 0, 1]  # 假设二分类任务，其中1表示正类，0表示负类


# 数据预处理函数
def preprocess_data(texts, labels, tokenizer, max_length=64):
    """
    对文本数据进行预处理并转换为模型输入格式

    参数:
   -texts: list of str, 文本数据
   -labels: list of int, 每个文本的标签
   -tokenizer: 预训练模型的Tokenizer
   -max_length: int, 序列的最大长度

    返回:
   -inputs: 字典格式的模型输入
   -label_tensor: 标签的张量表示
    """
    # Tokenization: 将文本转换为input_ids, attention_mask等格式
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt")

    # 将标签转换为张量格式
    label_tensor = torch.tensor(labels)

    return inputs, label_tensor


# 进行数据预处理
inputs, label_tensor = preprocess_data(texts, labels, tokenizer)

# 打印预处理结果
print("输入数据:", inputs)
print("标签张量:", label_tensor)

# 1. 组装为 PyTorch 的 TensorDataset
dataset = TensorDataset(
    inputs["input_ids"],
    inputs["attention_mask"],
    inputs["token_type_ids"],
    label_tensor,
)

# 2. 创建 DataLoader 进行批次加载（Batching）
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. 模拟在训练循环中提取 Batch 数据
for batch in dataloader:
    b_input_ids, b_attn_mask, b_token_type_ids, b_labels = batch
    print("\n--- 批次数据提取示例 ---")
    print("Batch input_ids 形状:", b_input_ids.shape)  # [2, 64]
    print("Batch labels 形状:", b_labels.shape)  # [2]
    break