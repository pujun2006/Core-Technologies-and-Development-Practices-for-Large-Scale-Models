import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载适配中文的 GPT-2 模型与分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 替换为中文预训练 GPT-2 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 显式指定 pad_token_id 以消除控制台警告
tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. 准备输入文本
input_text = "在科技发展的未来，"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids

# 3. 单次 Top-K 采样测试
top_k_value = 50
output = model.generate(
    input_ids,
    max_length=50,  # 最大生成长度
    top_k=top_k_value,  # 仅从概率最高的前 50 个候选词中采样
    do_sample=True,  # 开启随机采样
)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("--- 单次 Top-K=50 生成结果 ---")
print("生成的文本:", generated_text)
print("-" * 40)

# 4. 对比不同 Top-K 值对生成多样性的影响
top_k_values = [10, 50, 100]
generated_texts = []

for k in top_k_values:
    # 建议设置 random seed 以便观测规律，或者多运行几次体验多样性
    output = model.generate(input_ids, max_length=50, top_k=k, do_sample=True)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_texts.append((k, text))

# 打印不同 K 值生成的文本
print("\n--- 不同 Top-K 参数对比 ---")
for k, text in generated_texts:
    print(f"[Top-K = {k}]\n{text}\n")