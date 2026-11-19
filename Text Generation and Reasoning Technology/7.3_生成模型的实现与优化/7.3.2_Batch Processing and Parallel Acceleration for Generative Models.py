import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 确定运行设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载适配中文的 GPT 模型
model_name = "uer/gpt2-chinese-cluecorpussmall"
print(f"正在加载模型至 {device} ...")

# ⚠️ 关键设置 1：因 Batch 生成需要 Padding，必须显式指定 padding_side="left"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name)

# ⚠️ 关键设置 2：GPT-2 没有默认 pad_token，需手动将 eos_token 设为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)


# ==========================================
# 规范的批量生成函数
# ==========================================
def batch_generate_texts(
    prompts,
    max_new_tokens=60,
    top_k=50,
    top_p=0.9,
    temperature=0.7,
    batch_size=2,
):
    """批量生成文本，通过批处理实现 GPU 并行加速"""
    generated_texts = []

    # 按 batch_size 切割 prompts
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # 编码输入，构建 input_ids 和 attention_mask
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,  # 配合 padding_side="left" 自动左填充
            truncation=True,
        ).to(device)

        # 多卡 DataParallel 兼容处理
        raw_model = model.module if hasattr(model, "module") else model

        with torch.no_grad():
            output = raw_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # 传入 attention_mask，忽略 pad_token
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 精准解码：按各个 prompt 实际的真实输入长度对 output 进行精准切片
        for j, out_ids in enumerate(output):
            # 获取当前序列中非 pad 的真实输入长度
            input_len = inputs.attention_mask[j].sum().item()
            # 截取新生成的 Token
            gen_ids = out_ids[inputs.input_ids.shape[1] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_texts.append(batch_prompts[j] + text)

    return generated_texts


# ==========================================
# 运行测试与多卡封装演示
# ==========================================
if __name__ == "__main__":
    prompts = [
        "未来的人工智能技术将会如何发展？",
        "机器学习在医疗行业的应用",
        "智能城市如何提高生活质量？",
    ]

    # ⚠️ 多卡（Multi-GPU）正确封装方式：
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel...")
        model = torch.nn.DataParallel(model)

    print("\n--- 正在进行 Batch 批量生成 ---")
    results = batch_generate_texts(prompts, batch_size=2)

    for prompt, generated_text in zip(prompts, results):
        print(f"提示: {prompt}")
        print(f"生成的文本: {generated_text}")
        print("-" * 60)