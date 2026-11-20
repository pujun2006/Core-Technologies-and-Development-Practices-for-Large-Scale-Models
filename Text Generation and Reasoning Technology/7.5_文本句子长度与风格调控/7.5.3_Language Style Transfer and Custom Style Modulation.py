import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 推荐使用具备指令理解能力的微调模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto"
).to(device)


def transfer_text_style_fewshot(
    input_text, target_style, example_input, example_output
):
    """通过 Few-Shot (少样本示例) 强行让模型学会风格改写，彻底解决复读原句的问题。"""
    system_prompt = "你是一名语言改写专家。请参考示例，将【原始文本】改写为【目标风格】。绝对不能原封不动重复原句！"

    # 构建带有样例 (Example) 的完整上下文
    user_content = f"""【目标风格】：{target_style}

【改写示例】：
原始文本：今天天气很好，适合出门散步。
改写结果：{example_output}

【正式任务】：
原始文本：{input_text}
改写结果："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.85,  # 稍微提高随机性，鼓励词汇创新
            top_p=0.9,
            repetition_penalty=1.2,  # 惩罚重复原句的 Token
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    original_text = "人工智能在未来的日常生活中，将会扮演越来越重要的角色。"

    # 为不同风格配置专门的 Example（示范）
    styles_config = {
        "幽默调侃": {
            "style": "幽默自嘲、带有网络梗",
            "example": "好家伙，以后连跟人吵架、抢豆腐脑都得看 AI 的脸色了！",
        },
        "严谨学术": {
            "style": "学术论文体，使用专业术语",
            "example": "据观测，人工智能技术正以指数级渗透率重塑人类社会之日常行为范式。",
        },
        "消极悲观": {
            "style": "充满警惕、担忧与危机感",
            "example": "警惕吧！我们正在亲手将掌控生活的权利，拱手让给一堆冷冰冰的代码。",
        },
    }

    print(f"【原始文本】: {original_text}\n" + "=" * 60)

    for style_name, config in styles_config.items():
        res = transfer_text_style_fewshot(
            original_text,
            target_style=config["style"],
            example_input="人工智能在未来的日常生活中，将会扮演越来越重要的角色。",
            example_output=config["example"],
        )
        print(f"【{style_name}风格迁移结果】:\n{res}")
        print("-" * 60)