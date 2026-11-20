import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置运行设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载适配中文的 GPT 模型
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 3. 规范的重复控制生成函数
# ==========================================
def generate_with_repetition_control(
    prompt,
    max_new_tokens=80,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    temperature=0.7,
):
    """通过 repetition_penalty 与 no_repeat_ngram_size 控制生成文本的重复率。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 显式关闭梯度计算，省显存并加速
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # 使用 max_new_tokens 精准控制增量
            repetition_penalty=repetition_penalty,  # 重复惩罚系数
            no_repeat_ngram_size=no_repeat_ngram_size,  # 禁止 N-gram 重复
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 精准切片：截取新生成的 Token 避免重复解码 Prompt
    generated_ids = output[0][inputs.input_ids.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return prompt + generated_text


# ==========================================
# 4. 运行测试：观察不同参数组合效果
# ==========================================
if __name__ == "__main__":
    prompt = "未来的人工智能技术将会如何改变日常生活？"

    print("--- 基础测试 ---")
    controlled_text = generate_with_repetition_control(prompt)
    print("生成文本:\n", controlled_text)
    print("=" * 60)

    # 测试不同参数组合（注：建议 N-gram 至少设为 3）
    penalties = [1.1, 1.3]
    ngrams = [3, 4]

    for penalty in penalties:
        for ngram_size in ngrams:
            print(f"\n【重复惩罚: {penalty} | N-gram 限制: {ngram_size}】")
            text = generate_with_repetition_control(
                prompt,
                repetition_penalty=penalty,
                no_repeat_ngram_size=ngram_size,
            )
            print("生成文本:\n", text)
            print("-" * 60)