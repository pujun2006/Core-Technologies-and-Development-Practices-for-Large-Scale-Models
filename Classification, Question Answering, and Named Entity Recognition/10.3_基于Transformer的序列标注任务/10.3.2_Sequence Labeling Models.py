import numpy as np
import torch
import evaluate
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ----------------------------------------------------------------------
# 1. 加载预训练的 BERT 分词器和模型
# ----------------------------------------------------------------------
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=9)

# ----------------------------------------------------------------------
# 2. 读取并解析本地 CoNLL-2003 数据集
# ----------------------------------------------------------------------
# CoNLL-2003 标准 9 类 NER 标签
label_list = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


def read_conll_file(file_path):
    """从本地 txt 文件读取并解析 CoNLL 格式数据"""
    tokens_list, ner_tags_list = [], []
    tokens, ner_tags = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 遇到空行或文件头/段落分隔标志，代表一句话结束
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    tokens_list.append(tokens)
                    ner_tags_list.append(ner_tags)
                    tokens, ner_tags = [], []
                continue

            splits = line.split()
            word = splits[0]
            label = splits[-1]  # CoNLL-2003 格式最后一列为 NER 标签

            tokens.append(word)
            ner_tags.append(label2id.get(label, 0))

        if tokens:
            tokens_list.append(tokens)
            ner_tags_list.append(ner_tags)

    return {"tokens": tokens_list, "ner_tags": ner_tags_list}


# 定义 Dataset 特征映射属性
features = Features(
    {
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list)),
    }
)

# 加载本地的 txt 数据
dataset = DatasetDict(
    {
        "train": Dataset.from_dict(
            read_conll_file("conll2003_data/train.txt"), features=features
        ),
        "validation": Dataset.from_dict(
            read_conll_file("conll2003_data/validation.txt"), features=features
        ),
        "test": Dataset.from_dict(
            read_conll_file("conll2003_data/test.txt"), features=features
        ),
    }
)

# 加载 seqeval 评估指标
metric = evaluate.load("seqeval")


# ----------------------------------------------------------------------
# 3. 数据预处理（对齐子词 Token 与标签）
# ----------------------------------------------------------------------
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊字符（如 [CLS], [SEP]）填入 -100，PyTorch 计算交叉熵损失时会自动忽略
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 单词的首个子词 Token 保留真实标签
                label_ids.append(label[word_idx])
            else:
                # 单词的后续子词 Token 填入 -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 映射预处理函数到数据集
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


# ----------------------------------------------------------------------
# 4. 定义模型评估计算函数
# ----------------------------------------------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除 -100 的 padding 标签，提取真实的预测与目标值
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ----------------------------------------------------------------------
# 5. 设置训练参数与 DataCollator
# ----------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 每个 Epoch 评估一次
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

# 动态对齐序列长度的 Collator，防止序列不一致导致的维度错误
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# ----------------------------------------------------------------------
# 6. 初始化 Trainer 并训练/评估
# ----------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 开启微调
print(">>> 开始训练模型...")
trainer.train()

# 模型评估
print("\n>>> 开始在验证集上评估模型...")
eval_results = trainer.evaluate()
print(eval_results)


# ----------------------------------------------------------------------
# 7. 定义单句测试推理函数
# ----------------------------------------------------------------------
def test_sequence_labeling(text):
    model.eval()
    words = text.split()
    tokens = tokenizer(
        words, return_tensors="pt", is_split_into_words=True, truncation=True
    )

    # 将张量移动到模型当前所在设备 (CPU/GPU)
    inputs = {k: v.to(model.device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**inputs)

    predictions = torch.argmax(output.logits, dim=2)[0].cpu().numpy()
    input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    result = []
    for token_str, pred_id in zip(input_tokens, predictions):
        # 过滤 [CLS], [SEP], [PAD] 等特殊字符
        if token_str not in [
            tokenizer.cls_token,
            tokenizer.sep_token,
            tokenizer.pad_token,
        ]:
            label = id2label[pred_id]
            result.append(f"{token_str:<15} : {label}")
    return result


# 测试案例运行
if __name__ == "__main__":
    test_text = "John Doe is a software engineer at OpenAI in San Francisco."
    print("\n>>> 测试输入句子:", test_text)
    print("\n>>> 模型输出标注结果:")
    print("\n".join(test_sequence_labeling(test_text)))