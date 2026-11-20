import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 1. 设置国内镜像源并加载适合中文问答的预训练模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "uer/roberta-base-chinese-extractive-qa"

print(f"正在加载中文问答模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. 定义上下文与问题
context = """
人工智能（AI）是计算机科学的一个分支，涉及开发能够模拟人类智能行为的算法。近年来，AI在各个领域迅速发展，特别是在自然语言处理（NLP）、
图像识别和语音识别等方面。BERT模型是一个著名的NLP模型，广泛应用于情感分析、文本分类、机器翻译和问答系统等应用。
"""

question = "BERT模型广泛应用在哪些领域？"

# 3. 输入文本编码
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"][0]

# 4. 模型推理获取 Logits
with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits[0]  # Shape: (seq_len,)
end_logits = outputs.end_logits[0]  # Shape: (seq_len,)

# 5. 【核心修复】：获取 Top-K 索引与得分并同时迭代
top_k = 3
start_top_scores, start_top_indices = torch.topk(start_logits, top_k)
end_top_scores, end_top_indices = torch.topk(end_logits, top_k)

candidate_answers = []

# 【核心修复】：同时遍历 Top-K 的得分与索引
for s_score, start_idx in zip(start_top_scores, start_top_indices):
    for e_score, end_idx in zip(end_top_scores, end_top_indices):
        # 确保结束位置在起始位置之后，且限制答案长度不超过 30 个 Token
        if end_idx >= start_idx and (end_idx - start_idx) < 30:
            answer_tokens = input_ids[start_idx: end_idx + 1]
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # 清理中文字符间可能产生的空格
            answer_text = answer_text.replace(" ", "")

            if answer_text:
                # 【核心修复】：语法与得分计算
                combined_score = (s_score.item() + e_score.item()) / 2
                candidate_answers.append((answer_text, round(combined_score, 4)))

# 6. 根据得分排序
sorted_answers = sorted(candidate_answers, key=lambda x: x[1], reverse=True)

# 7. 输出结果
print("\n问题:", question)
if sorted_answers:
    print("最佳答案:", sorted_answers[0][0])
    print("\n所有候选答案:")
    for answer, score in sorted_answers:
        print(f"答案: {answer}, 综合得分: {score}")
else:
    print("未匹配到有效答案。")