import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置运行设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载适配中文的 GPT 模型
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 防御性设置：防止缺少 pad_token 产生控制台警告
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 3. 规范的 Temperature 探针生成函数
# ==========================================
def generate_with_temperature(
    prompt, temperature=1.0, max_new_tokens=60, top_k=50, top_p=0.9
):
    """通过设定温度参数调节生成的多样性和确定性。"""
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 显式关闭梯度计算，节省显存并加快速度
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # 使用 max_new_tokens 准确控制生成增量
            temperature=temperature,  # 控制生成温度
            top_k=top_k,
            top_p=top_p,
            do_sample=True,  # 开启采样模式
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 精准切片：只截取模型新生成的内容，不重复解码原始 Prompt
    generated_ids = output[0][inputs.input_ids.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return prompt + generated_text


# ==========================================
# 4. 测试不同 Temperature 对生成的实际效果
# ==========================================
if __name__ == "__main__":
    prompt = "未来的人工智能技术将如何影响日常生活？"
    temperatures = [0.2, 0.7, 1.0, 1.4]

    print(f"原始 Prompt: {prompt}\n" + "=" * 60)

    for temp in temperatures:
        text = generate_with_temperature(prompt, temperature=temp)
        print(f"【Temperature = {temp}】")
        print(f"生成文本: {text}")
        print("-" * 60)