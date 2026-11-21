import evaluate
import numpy as np
import requests
import torch
import torch.nn as nn
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

# ======================================================================
# 1. 加载本地 CoNLL-2003 数据集与模型初始化
# ======================================================================
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model = BertForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)


def read_conll_file(file_path):
    """读取并解析本地 CoNLL 格式文件"""
    tokens_list, ner_tags_list = [], []
    tokens, ner_tags = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    tokens_list.append(tokens)
                    ner_tags_list.append(ner_tags)
                    tokens, ner_tags = [], []
                continue

            splits = line.split()
            word = splits[0]
            label = splits[-1]

            tokens.append(word)
            ner_tags.append(label2id.get(label, 0))

        if tokens:
            tokens_list.append(tokens)
            ner_tags_list.append(ner_tags)

    return {"tokens": tokens_list, "ner_tags": ner_tags_list}


features = Features(
    {
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list)),
    }
)

# 读取本地数据文件构建 DatasetDict
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

metric = evaluate.load("seqeval")


# ======================================================================
# 2. 数据预处理（子词 Token 与标签对齐）
# ======================================================================
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


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ======================================================================
# 3. 稳健的类别权重计算与自定义 Weighted Loss Trainer
# ======================================================================
# 1) 提取所有非 -100 的训练标签
train_labels = [
    label
    for example in tokenized_datasets["train"]["labels"]
    for label in example
    if label != -100
]

# 2) 提取训练集中实际出现的唯一标签 ID
unique_labels = np.unique(train_labels)

# 3) 计算实际存在类别的权重
computed_weights = compute_class_weight(
    "balanced", classes=unique_labels, y=train_labels
)

# 4) 创建全量 (9类) 权重数组，默认平滑值为 1.0
class_weights_np = np.ones(len(label_list), dtype=np.float32)
for label_id, weight in zip(unique_labels, computed_weights):
    class_weights_np[label_id] = weight

class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print(">>> 类别权重初始化完成:\n", class_weights)


# 自定义 Trainer 以应用加权交叉熵损失 (Weighted CrossEntropyLoss)
class WeightedLossTrainer(Trainer):

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 将权重向量自动推送到模型所在的设备 (CPU/GPU)
        weight = class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


# ======================================================================
# 4. 训练配置与模型微调
# ======================================================================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=3e-5,
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(
        predictions=true_predictions, references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 使用支持动态填充的 Collator 避免维度不匹配
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("\n>>> 开始微调 NER 模型...")
trainer.train()

# ======================================================================
# 5. 上下文追踪器与 Wikipedia 知识图谱查询服务
# ======================================================================
class MultiTurnContextTracker:

    def __init__(self):
        self.context = []

    def update_context(self, text, entities):
        entry = {"text": text, "entities": entities}
        self.context.append(entry)

    def get_context(self):
        return self.context


class KnowledgeGraph:

    def get_info(self, entity):
        formatted_entity = entity.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_entity}"
        headers = {
            "User-Agent": "NER_KnowledgeGraph_Bot/1.0 (contact@example.com)"
        }
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json().get("extract", "未找到摘要信息。")
            return "没有找到相关知识库信息。"
        except Exception as e:
            return f"网络查询异常: {str(e)}"


context_tracker = MultiTurnContextTracker()
knowledge_graph = KnowledgeGraph()

# 采用 pipeline 实现长短实体自动聚合与子词还原
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)


def extract_entities(text):
    """提取完整的命名实体及类别"""
    results = ner_pipeline(text)
    entities = []
    for item in results:
        entities.append((item["word"].strip(), item["entity_group"]))
    return entities


# ======================================================================
# 6. 多轮交互对话测试
# ======================================================================
def interactive_test():
    print("\n" + "=" * 50)
    print("多轮对话系统已启动，输入 '退出' 或 'exit' 以结束对话。")
    print("=" * 50)

    while True:
        user_input = input("\n用户: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["退出", "exit"]:
            print("对话结束。")
            break

        entities = extract_entities(user_input)
        context_tracker.update_context(user_input, entities)

        print("\n[识别到的实体]:", entities if entities else "未识别到命名实体")

        for entity_text, entity_label in entities:
            if entity_label in ["PER", "LOC", "ORG", "MISC"]:
                info = knowledge_graph.get_info(entity_text)
                print(f"-> 知识图谱 [{entity_text}] ({entity_label}): {info}")


if __name__ == "__main__":
    interactive_test()