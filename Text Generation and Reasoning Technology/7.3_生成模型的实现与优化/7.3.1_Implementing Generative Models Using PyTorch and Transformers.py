import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载适配中文的 GPT-2 模型与分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 防御性配置：设置 pad_token_id 消除控制台警告
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token_id = tokenizer.eos_token_id


# 2. 封装标准的文本生成函数
def generate_text(
    prompt,
    max_new_tokens=80,  # 推荐使用 max_new_tokens 控制生成增量，比 max_length 更安全
    top_k=50,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.1,
):
    """使用 GPT 模型生成文本，设置生成参数以控制生成的内容多样性和质量。"""
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    # 显式使用 no_grad 避免梯度占用显存，提升推理速度
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,  # 控制新生成词的数量
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,  # 惩罚重复词
            do_sample=True,  # 开启采样模式
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 精准切片：只提取模型新生成的文本内容，避免重复拼接输入 Prompt
    generated_ids = output[0][input_ids.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return prompt + generated_text  # 将原始 Prompt 与新生成文本完整拼接输出


# ==========================================
# 3. 运行单次与批量测试
# ==========================================
# 单次生成测试
prompt = "在人工智能的未来发展中，"
generated_text = generate_text(prompt)
print("--- 单次生成结果 ---")
print(generated_text)
print("=" * 60)

# 批量提示词生成测试
prompts = [
    "在未来的医疗技术中，",
    "随着科技的进步，",
    "人工智能将如何影响教育？",
]

print("\n--- 批量生成结果 ---")
for p in prompts:
    result = generate_text(
        p, max_new_tokens=60, top_k=40, top_p=0.85, temperature=0.8
    )
    print(f"提示: {p}")
    print(f"生成的文本: {result}")
    print("-" * 60)