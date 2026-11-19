import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载适配中文的 GPT-2 模型与分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 替换为中文 GPT-2 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 显式指定 pad_token_id 以消除控制台警告
tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. 准备输入文本
input_text = "未来的科技发展将会带来"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids

# 3. 单次 Top-P (Nucleus) 采样生成
top_p_value = 0.9

with torch.no_grad():  # 开启无梯度推理上下文，节省显存
    output = model.generate(
        input_ids,
        max_length=50,  # 最大生成长度
        top_p=top_p_value,  # 设置 Top-P (核采样) 累积概率阈值
        do_sample=True,  # 开启随机采样
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("--- 单次 Top-P=0.9 生成结果 ---")
print("生成的文本:", generated_text)
print("-" * 40)

# 4. 对比不同 Top-P 值对生成结果的影响
top_p_values = [0.7, 0.9, 0.95]
generated_texts = []

with torch.no_grad():
    for p in top_p_values:
        output = model.generate(
            input_ids,
            max_length=50,
            top_p=p,
            do_sample=True,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append((p, text))

# 打印不同 P 值生成的文本
print("\n--- 不同 Top-P 参数对比 ---")
for p, text in generated_texts:
    print(f"[Top-P = {p}]\n{text}\n")