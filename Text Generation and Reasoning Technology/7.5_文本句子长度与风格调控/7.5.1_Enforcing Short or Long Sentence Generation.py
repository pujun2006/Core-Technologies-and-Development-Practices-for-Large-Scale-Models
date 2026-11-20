import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 优先推荐轻量强力模型 Qwen2.5-0.5B，如果显存/网络有限可换回小模型
# model_name = "uer/gpt2-chinese-cluecorpussmall"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 强烈推荐：仅约1GB显存，生成质量剧增

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto"
).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 文本清洗工具：彻底消除汉字之间的异常空格
# ==========================================
def clean_chinese_spaces(text):
    """去除中文汉字与汉字、中文标点之间的多余空格。"""
    # 匹配汉字/中文标点之间的空格并删除
    text = re.sub(
        r"(?<=[\u4e00-\u9fa5，。！？；：（）《》])\s+(?=[\u4e00-\u9fa5，。！？；：（）《》])",
        "",
        text,
    )
    return text.strip()


# ==========================================
# 优化后的生成函数：注入重复惩罚与高质量采样
# ==========================================
def generate_high_quality(
    prompt, min_new_tokens=5, max_new_tokens=50, temperature=0.7
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.8,  # 核采样：剔除低概率词，提升通顺度
            repetition_penalty=1.2,  # 重复惩罚：抑制死循环
            no_repeat_ngram_size=3,  # 硬限制：禁止连续3词重复
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 精准提取新生成部分
    generated_ids = output[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 进行中文格式清洗
    cleaned_generated = clean_chinese_spaces(generated_text)

    return prompt + cleaned_generated


# ==========================================
# 动态长度生成测试
# ==========================================
def dynamic_length_generation(prompt, length_type="short"):
    config = {
        "short": (5, 20),
        "medium": (20, 50),
        "long": (50, 100),
    }
    min_tok, max_tok = config.get(length_type, (20, 50))
    return generate_high_quality(
        prompt, min_new_tokens=min_tok, max_new_tokens=max_tok
    )


if __name__ == "__main__":
    prompt = "人工智能在未来将会"

    print("--- 优化后的动态长度生成测试 ---")
    for l_type in ["short", "medium", "long"]:
        res = dynamic_length_generation(prompt, length_type=l_type)
        print(f"【{l_type.upper()} 结果】:")
        print(res)
        print("-" * 50)