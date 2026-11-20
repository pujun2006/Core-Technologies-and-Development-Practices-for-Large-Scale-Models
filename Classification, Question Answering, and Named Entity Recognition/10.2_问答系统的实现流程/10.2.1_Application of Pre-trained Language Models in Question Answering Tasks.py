import os

# 【核心修复】：必须在导入 transformers/torch 之前配置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 1. 使用主流且公开可访问的中文问答模型标识符
model_name = "uer/roberta-base-chinese-extractive-qa"

print(f"正在从 Hugging Face 镜像源加载模型: {model_name} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("模型加载成功！")
except Exception as e:
    print(f"加载失败，请检查网络。错误信息: {e}")
    exit(1)

# 2. 定义中文上下文与问题
context = """
机器学习是人工智能的一个分支，利用数据和算法自动执行复杂任务。深度学习作为机器学习的一部分，
通过神经网络模拟人脑的结构和功能。BERT模型是一种双向Transformer模型，通过在上下文中了解单词之间的关系，能够更准确地完成问答、分类等任务。
在NLP领域，BERT被广泛应用于情感分析、文本分类、问答系统等应用。BERT模型通过大规模的预训练数据进行训练，具有很高的语言理解能力。
"""

question = "BERT模型用于完成哪些任务？"

# 3. 编码与推理
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"][0]

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]

# 4. 获取前 Top-K 候选答案
top_k = 3
start_top_scores, start_top_indices = torch.topk(start_logits, top_k)
end_top_scores, end_top_indices = torch.topk(end_logits, top_k)

candidate_answers = []

for s_score, start_idx in zip(start_top_scores, start_top_indices):
    for e_score, end_idx in zip(end_top_scores, end_top_indices):
        # 确保结束位置在起始位置之后，且限制答案最大长度
        if end_idx >= start_idx and (end_idx - start_idx) < 30:
            answer_tokens = input_ids[start_idx : end_idx + 1]
            answer_text = tokenizer.decode(
                answer_tokens, skip_special_tokens=True
            )

            # 过滤掉首尾空格
            answer_text = answer_text.replace(" ", "")
            if answer_text:
                avg_score = (s_score.item() + e_score.item()) / 2
                candidate_answers.append({
                    "answer": answer_text,
                    "start_score": round(s_score.item(), 4),
                    "end_score": round(e_score.item(), 4),
                    "avg_score": round(avg_score, 4),
                })

# 按综合得分从高到低排序
candidate_answers = sorted(
    candidate_answers, key=lambda x: x["avg_score"], reverse=True
)

# 5. 输出结果
print("\n问题:", question)
if candidate_answers:
    print("最佳答案:", candidate_answers[0]["answer"])
    print("\n候选答案列表:")
    for ans in candidate_answers:
        print(
            f"答案: {ans['answer']} | 起始得分: {ans['start_score']} | 结束得分: {ans['end_score']} | 综合得分: {ans['avg_score']}"
        )
else:
    print("未找到有效答案。")