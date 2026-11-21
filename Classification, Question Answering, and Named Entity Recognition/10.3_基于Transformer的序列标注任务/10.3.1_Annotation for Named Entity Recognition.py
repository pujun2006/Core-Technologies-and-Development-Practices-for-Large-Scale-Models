import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

# ====================================================
# 1. 标签列表与标签映射定义
# ====================================================
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
label2id = {label: i for i, label in enumerate(label_list)}

# ====================================================
# 2. 内置示范数据集（按 CoNLL-2003 格式构建，不依赖网络下载）
# ====================================================
dummy_data = {
    "train": {
        "tokens": [
            ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
            ["Peter", "Blackburn", "reports", "from", "BRUSSELS", "."],
            ["Tesla", "Inc.", "was", "founded", "in", "Austin", ",", "Texas", "."],
        ],
        "ner_tags": [
            [label2id["B-ORG"], label2id["O"], label2id["B-MISC"], label2id["O"], label2id["O"], label2id["O"], label2id["B-MISC"], label2id["O"], label2id["O"]],
            [label2id["B-PER"], label2id["I-PER"], label2id["O"], label2id["O"], label2id["B-LOC"], label2id["O"]],
            [label2id["B-ORG"], label2id["I-ORG"], label2id["O"], label2id["O"], label2id["O"], label2id["B-LOC"], label2id["O"], label2id["B-LOC"], label2id["O"]],
        ]
    },
    "validation": {
        "tokens": [
            ["CRICKET", "-", "LEICESTERSHIRE", "TAKE", "OVER", "AT", "TOP", "."],
            ["Elon", "Musk", "visited", "London", "yesterday", "."],
        ],
        "ner_tags": [
            [label2id["O"], label2id["O"], label2id["B-ORG"], label2id["O"], label2id["O"], label2id["O"], label2id["O"], label2id["O"]],
            [label2id["B-PER"], label2id["I-PER"], label2id["O"], label2id["B-LOC"], label2id["O"], label2id["O"]],
        ]
    },
    "test": {
        "tokens": [
            ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", "."],
            ["Google", "is", "located", "in", "California", "."],
        ],
        "ner_tags": [
            [label2id["O"], label2id["O"], label2id["B-LOC"], label2id["O"], label2id["O"], label2id["O"], label2id["O"]],
            [label2id["B-ORG"], label2id["O"], label2id["O"], label2id["O"], label2id["B-LOC"], label2id["O"]],
        ]
    }
}

dataset = DatasetDict({
    "train": Dataset.from_dict(dummy_data["train"]),
    "validation": Dataset.from_dict(dummy_data["validation"]),
    "test": Dataset.from_dict(dummy_data["test"])
})

print("数据集创建成功！示例：", dataset["train"][0])

# ====================================================
# 3. 加载预训练模型、分词器与评估指标
# ====================================================
# 加载 NER 评估器
metric = evaluate.load("seqeval")

# 加载 NER 模型与分词器
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name, num_labels=len(label_list))

# 加载 QA 多轮问答模型与分词器
qa_model_name = "google/flan-t5-base"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

# ====================================================
# 4. 数据预处理与标签对齐函数
# ====================================================
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
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 对数据集做映射预处理
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ====================================================
# 5. 训练参数设置与评估指标计算函数
# ====================================================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 最新 transformers 推荐使用 eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=1,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ====================================================
# 6. 构造 Trainer 并执行训练
# ====================================================
# 关键修复：实例化 DataCollatorForTokenClassification 进行动态 Padding
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=ner_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # 传入 Collator 处理变长序列
)

print("\n--- 开始训练 NER 模型 ---")
trainer.train()

# ====================================================
# 7. 测试集预测与结果打印
# ====================================================
print("\n--- 测试集预测评估 ---")
test_results = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(test_results.predictions, axis=2)

print("\n--- NER 预测样例展示 ---")
num_samples = min(5, len(tokenized_datasets["test"]))
for i in range(num_samples):
    tokens = tokenized_datasets["test"][i]["tokens"]
    # 同时过滤 -100 保持位置严密对齐
    labels = [
        label_list[l]
        for l in test_results.label_ids[i]
        if l != -100
    ]
    preds = [
        label_list[p]
        for p, l in zip(predictions[i], test_results.label_ids[i])
        if l != -100
    ]

    print(f"Tokens:      {' '.join(tokens)}")
    print(f"Labels:      {labels}")
    print(f"Predictions: {preds}")
    print("-" * 50)

# ====================================================
# 8. 构建 Pipeline 管道与多轮 QA 上下文维护
# ====================================================
# ====================================================
# 8. 构建 Pipeline 管道与多轮 QA 上下文维护（标准修复版）
# ====================================================
# 1. 增加 aggregation_strategy="simple"，解决 WordPiece 拆词问题
ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

context_history = []


def query_knowledge_graph(question, history):
    # 步骤 A: 抽取实体
    entities = ner_pipeline(question)
    print(f"\n[用户问题]: {question}")
    print("识别出的实体:")
    if entities:
        for entity in entities:
            # aggregation_strategy="simple" 后，字段名从 'entity' 变为了 'entity_group'
            print(f"  - {entity['word']}: {entity['entity_group']} (置信度: {entity['score']:.2f})")
    else:
        print("  - 无")

    # 步骤 B: 拼接上下文历史
    history.append(question)
    long_context = " ".join(history[-5:])
    prompt = f"Answer the question based on context. Context: {long_context} Question: {question}"

    # 步骤 C: 正确地使用 T5 (Seq2Seq) 生成答案
    inputs = qa_tokenizer(prompt, return_tensors="pt").to(qa_model.device)

    # 显式生成，避免 text-generation 警告
    outputs = qa_model.generate(
        **inputs,
        max_new_tokens=50
    )

    answer_text = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("生成回答:")
    print(f"  {answer_text}")
    return answer_text


# 运行测试示例
print("\n--- 开始多轮对话测试 ---")
query_knowledge_graph("Who is the CEO of Tesla?", context_history)
query_knowledge_graph("Where is the headquarters of Tesla?", context_history)