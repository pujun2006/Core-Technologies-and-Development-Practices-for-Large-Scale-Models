import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载中文 GPT-2 模型与分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 显式指定 pad_token_id 消除警告
tokenizer.pad_token_id = tokenizer.eos_token_id

input_text = "在未来的城市里，科技的发展让生活发生了巨大的变化。"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# ==========================================
# 方式 1：一次性生成长文本（推荐使用采样模式）
# ==========================================
# 💡 长文本推荐 Top-P / Top-K + Temperature 组合，避免 Beam Search 的单调和死锁
with torch.no_grad():
    output_one_shot = model.generate(
        input_ids,
        max_length=200,
        do_sample=True,  # 开启采样
        top_p=0.9,  # 核采样，保持流畅度与创造力
        temperature=0.8,  # 控制生成多样性
        no_repeat_ngram_size=2,  # 适度抑制重复（2-gram 比 3-gram 更自然）
    )

generated_text_one_shot = tokenizer.decode(
    output_one_shot[0], skip_special_tokens=True
)
print("--- 方式 1：一次性生成的长文本 ---")
print(generated_text_one_shot)
print("-" * 50)


# ==========================================
# 方式 2：规范的分段迭代扩展生成（Token 级精准切片）
# ==========================================
current_ids = input_ids  # 使用 Tensor 进行 Token 级追踪，避免字符串切片错位
step_generate_tokens = 30  # 每次新增生成的 Token 数量

with torch.no_grad():
    for _ in range(3):  # 迭代扩展 3 次
        input_len = current_ids.shape[1]

        # 每次在原有 Token 序列基础上，增加指定数量的新 Token
        output_step = model.generate(
            current_ids,
            max_length=input_len + step_generate_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )

        # ✅ 正确做法：直接在 Token 层面切片，只保留本次新生成的 Token ID，再更新 current_ids
        current_ids = output_step

# 最终一次性将全部 Token 序列解码为文本
final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)

print("\n--- 方式 2：分段迭代扩展生成的长文本 ---")
print(final_text)