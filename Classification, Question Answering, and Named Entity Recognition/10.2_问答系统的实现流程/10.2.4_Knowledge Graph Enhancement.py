import os
import torch
from py2neo import Graph
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 1. 设置国内镜像并加载中文问答模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "uer/roberta-base-chinese-extractive-qa"

print(f"正在加载中文问答模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. 连接 Neo4j 数据库（若无需真实运行 Neo4j，可使用 Dummy fallback）
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    graph.run("RETURN 1")
    neo4j_available = True
except Exception as e:
    print(
        "Warning: 无法连接到 Neo4j 数据库，将启用模拟图谱数据进行演示。",
        e,
    )
    neo4j_available = False

# 模拟知识图谱数据（当 Neo4j 未连接时自动切换）
dummy_kg = {
    "Transformer": ["BERT", "GPT", "T5", "注意力机制"],
    "自然语言处理": ["机器翻译", "文本生成", "情感分析"],
    "情感分析": ["词性标注", "文本分类", "深度学习"],
}


# 3. 规范化知识图谱查询函数
def query_knowledge_graph(entity):
    if not entity or len(entity) > 15:  # 实体名过长通常是句子，直接过滤
        return []

    entity = entity.strip()

    if neo4j_available:
        # 使用参数化查询防止 Cypher 注入
        cypher = """
        MATCH (e)-[r]->(related)
        WHERE e.name = $entity_name
        RETURN related.name as name
        LIMIT 5
        """
        try:
            results = graph.run(cypher, entity_name=entity)
            return [rec["name"] for rec in results]
        except Exception:
            return []
    else:
        # 从模拟数据查询
        return dummy_kg.get(entity, [])


# 4. 抽取式问答核心函数（包含 Top-K 过滤与停用词过滤）
def extract_answer(model, tokenizer, question, context, top_k=5):
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"][0]

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_scores, start_indices = torch.topk(start_logits, top_k)
    end_scores, end_indices = torch.topk(end_logits, top_k)

    stop_words = {"和", "等", "的", "包括", "与", "及", "是", "如"}
    candidates = []

    for s_score, s_idx in zip(start_scores, start_indices):
        for e_score, e_idx in zip(end_scores, end_indices):
            s_i, e_i = s_idx.item(), e_idx.item()
            if e_i >= s_i and 1 <= (e_i - s_i + 1) <= 25:
                tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[s_i : e_i + 1]
                )
                text = (
                    tokenizer.convert_tokens_to_string(tokens)
                    .replace(" ", "")
                    .replace("##", "")
                )

                if text and text not in stop_words:
                    if not any(
                        text.startswith(sw) for sw in ["和", "等", "的"]
                    ):
                        score = s_score.item() + e_score.item()
                        candidates.append((text, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else ""


# 5. 知识图谱增强问答流程
def knowledge_enhanced_qa(model, tokenizer, context, questions):
    base_context = context.strip()
    history = base_context

    for i, question in enumerate(questions):
        print(f"===== 第 {i + 1} 轮问答 =====")

        # 1) 从文本中抽取基础答案
        raw_answer = extract_answer(model, tokenizer, question, history)
        if not raw_answer:
            raw_answer = "未在上下文中找到明确答案"

        # 2) 知识图谱检索增强 (KG Search)
        kg_info = query_knowledge_graph(raw_answer)

        # 3) 组合生成最终回答
        final_answer = raw_answer
        if kg_info:
            final_answer += (
                f"（[知识图谱补充] 相关实体节点：{', '.join(kg_info)}）"
            )

        # 4) 更新上下文历史
        history += f" 问：{question} 答：{raw_answer}"

        print(f"问题: {question}")
        print(f"回答: {final_answer}\n")


# 运行测试
context = """
Transformer模型是一种深度学习架构，广泛应用于自然语言处理任务，如机器翻译、文本生成和情感分析。该模型通过注意力机制来识别句子中的重要信息。
"""

questions = [
    "Transformer模型的主要功能是什么？",
    "该模型的应用领域有哪些？",
]

knowledge_enhanced_qa(model, tokenizer, context, questions)