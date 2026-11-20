import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 1. 配置国内镜像源，替换为支持中文抽取式问答的模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "uer/roberta-base-chinese-extractive-qa"

print(f"正在加载中文问答模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. 定义上下文与多轮问题
context = """
机器学习在多个领域得到了广泛应用，特别是在自然语言处理、图像识别、医疗健康、金融等方面。
Transformer模型作为一种深度学习架构，极大地提升了文本处理的效率和效果，包括BERT和GPT等著名模型。
"""

# 注意：抽取式问答模型只能回答原文中明确给出的信息
questions = [
    "机器学习应用在哪些领域？",
    "有哪些著名的Transformer模型？",
    "Transformer模型有什么作用？",
]


def multi_turn_qa(model, tokenizer, context, questions):
    # 保存基础上下文与历史问答片段
    base_context = context.strip()
    qa_history = ""

    for i, question in enumerate(questions):
        print(f"===== 第 {i + 1} 轮问答 =====")

        # 拼接历史上下文（规范格式）
        full_context = base_context + qa_history

        # 截断输入以防止超过 BERT 的 512 token 限制
        inputs = tokenizer(
            question,
            full_context,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]

        # 模型前向传播
        with torch.no_grad():
            outputs = model(**inputs)

        # 降到一维向量，防止索引计算错误
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # 获取得分最高的位置索引
        answer_start = torch.argmax(start_logits).item()
        answer_end = torch.argmax(end_logits).item()

        # 校验索引有效性（起始位置必须小于等于结束位置，且不能超过序列长度）
        if answer_end >= answer_start and (answer_end - answer_start) < 50:
            answer_tokens = input_ids[answer_start : answer_end + 1]
            answer = tokenizer.decode(
                answer_tokens, skip_special_tokens=True
            ).replace(" ", "")
        else:
            answer = "未在原文中找到合适答案"

        if not answer:
            answer = "未找到有效答案"

        # 更新历史对答（供下一轮参考）
        qa_history += f" 问：{question} 答：{answer}"

        print(f"问题: {question}")
        print(f"回答: {answer}\n")


# 执行多轮问答
multi_turn_qa(model, tokenizer, context, questions)