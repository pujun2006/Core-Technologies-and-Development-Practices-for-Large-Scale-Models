import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型与分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 适配中文的 GPT2 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置 pad_token 防止控制台打印警告
tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. 准备输入文本
input_text = "在未来的人工智能领域，"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids

# 3. 设置 Beam Search 参数
beam_size = 5
num_return_sequences = 3

# 4. 使用高层 API 直接生成（推荐方式）
outputs = model.generate(
    input_ids,
    max_length=50,  # 生成的最大 token 长度
    num_beams=beam_size,  # 设置束宽 (Beam Size)
    num_return_sequences=num_return_sequences,  # 指定返回得分最高的序列数量
    early_stopping=True,  # 达到 EOS 时提前结束
    length_penalty=1.0,  # 长度惩罚系数（>1 倾向长句，<1 倾向短句）
)

# 5. 解码并打印输出结果
print("--- Beam Search 输出结果 ---")
for i, sequence in enumerate(outputs):
    text = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"Beam {i+1}: {text}")