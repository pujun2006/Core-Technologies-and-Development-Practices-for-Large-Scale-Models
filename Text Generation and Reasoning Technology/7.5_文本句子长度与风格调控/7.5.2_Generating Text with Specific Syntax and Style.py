import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载适配中文的指令微调模型（Instruct Model）
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto"
).to(device)


# ==========================================
# 2. 规范的风格控制生成函数（基于 System Prompt）
# ==========================================
def generate_with_style(
    prompt, style_instruction, max_new_tokens=100, temperature=0.7, top_p=0.9
):
    """通过系统提示词 (System Prompt) 严格控制大模型的生成风格与格式。"""
    # 构建标准的对话消息列表 (Chat Messages)
    messages = [
        {"role": "system", "content": style_instruction},  # 定义风格角色规则
        {"role": "user", "content": prompt},  # 具体的生成任务
    ]

    # 使用分词器自带的 Chat Template 转换为模型规范的输入格式
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    # 显式禁用梯度加速推理
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,  # 防止死循环重复
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 精准截取增量生成的文本
    generated_ids = output[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ==========================================
# 3. 运行风格测试
# ==========================================
if __name__ == "__main__":
    task_prompt = "请阐述人工智能技术对未来社会的影响。"

    # 定义不同的系统风格指令
    news_style = "你是一名资深科技新闻记者。请用严肃、客观、符合新闻报道规范的语言风格回答，并加上拟定的新闻标题。"
    novel_style = "你是一名科幻小说家。请用带有强烈画面感、小说叙事体裁和引人入胜的情节风格来展现这个主题。"
    academic_style = "你是一名计算机科学教授。请用严谨的学术论文风格回答，使用标准的学术术语和逻辑结构。"

    print("--- 新闻风格测试 ---")
    print(generate_with_style(task_prompt, news_style, max_new_tokens=120))
    print("\n" + "=" * 60 + "\n")

    print("--- 小说风格测试 ---")
    print(generate_with_style(task_prompt, novel_style, max_new_tokens=120))
    print("\n" + "=" * 60 + "\n")

    print("--- 学术风格测试 ---")
    print(
        generate_with_style(task_prompt, academic_style, max_new_tokens=120)
    )