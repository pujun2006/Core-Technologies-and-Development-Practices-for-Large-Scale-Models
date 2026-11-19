import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# 1. 定义模型名称与加载 Tokenizer
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 示例文本数据
texts = [
    "自然语言处理是一门研究计算机理解和生成人类语言的学科。",
    "Transformer模型是目前最流行的自然语言处理模型之一。",
    "Hugging Face的分词工具支持多种语言模型的分词。",
]
labels = [1, 1, 0]  # 示例分类标签


# 3. 构建自定义 PyTorch Dataset 封装
class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 单条文本编码（暂不强制 padding="max_length"，留给 DataLoader 动态 Padding）
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 压缩多余的 Batch 维度，确保每个 item 是 (L,) 形状
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# 4. 优化点：使用 DataCollatorWithPadding 实现动态 Padding（根据当前 Batch 最长句子填充，节省显存与计算力）
def collate_fn(batch):
    # 提取 batch 中的单条输入字典
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])

    # 使用 tokenizer.pad 进行对齐填充
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt",
    )

    padded["labels"] = labels
    return padded


# 5. 创建 Dataset 与 DataLoader
dataset = TextClassificationDataset(
    texts, labels, tokenizer, max_length=64
)
dataloader = DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

# 6. 验证输出与解码（解码可直观查看 [CLS], [SEP], [PAD] 结构）
print("=== Batch 动态填充与解码验证 ===\n")
for step, batch in enumerate(dataloader):
    print(f"--- Batch {step + 1} ---")
    print(f"input_ids 形状: {batch['input_ids'].shape}")
    print(f"attention_mask 形状: {batch['attention_mask'].shape}")
    print(f"labels 张量: {batch['labels']}")

    # 还原显示文本（验证 [CLS], [SEP] 和 [PAD]）
    for i in range(len(batch["input_ids"])):
        decoded_text = tokenizer.decode(
            batch["input_ids"][i], skip_special_tokens=False
        )
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
        print(f"\n[样本 {i + 1} 还原文本]: {decoded_text}")
        print(f"[样本 {i + 1} Token 序列]: {tokens[:12]}...")  # 展示前12个 token
    print("\n" + "=" * 50)