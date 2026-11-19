import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 加载模型与分词器（使用中文预训练 GPT 模型）
# ==========================================
model_name = "uer/gpt2-chinese-cluecorpussmall"
print(f"正在加载模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --------------------------------------------------
# 关键修复：防御性设置 eos_token 和 pad_token，防止 NoneType 报错
# --------------------------------------------------
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"

tokenizer.pad_token_id = tokenizer.eos_token_id


# ==========================================
# 2. 多轮对话生成函数（基于 Token 级准确拼接与窗口截断）
# ==========================================
def generate_response_multiturn(
    input_text, chat_history_ids=None, max_context_length=256, max_new_tokens=50
):
    """
    input_text: 当前用户输入的文本
    chat_history_ids: 拼接了历史对话的 Token Tensor
    max_context_length: 允许输入模型的最大上下文 Token 长度（滑动窗口保护）
    max_new_tokens: 本轮生成的新 Token 最大数量
    """
    # 1. 安全拼接 eos_token（防止 tokenizer.eos_token 为 None 导致报错）
    eos_str = tokenizer.eos_token if tokenizer.eos_token else ""
    new_user_input_ids = tokenizer.encode(input_text + eos_str, return_tensors="pt")

    # 2. 将当前输入与历史对话 Token 拼接
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # 3. 滑动窗口保护：超过最大上下文长度时，仅保留最近的 Token，避免显存溢出
    if bot_input_ids.shape[-1] > max_context_length:
        bot_input_ids = bot_input_ids[:, -max_context_length:]

    # 4. 调用模型生成回复
    with torch.no_grad():
        chat_history_ids = model.generate(
            bot_input_ids,
            max_new_tokens=max_new_tokens,         # 明确控制生成的新词数量
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,                        # 开启采样模式
            top_p=0.9,                             # 核采样
            temperature=0.7,                       # 控制采样温度
            no_repeat_ngram_size=2,                # 避免局部重复词
        )

    # 5. 精准截取本次生成的 Token（避开输入的 prompt）
    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return response, chat_history_ids


# ==========================================
# 3. 运行多轮对话测试
# ==========================================
if __name__ == "__main__":
    user_inputs = [
        "你好！今天的天气怎么样？",
        "那么明天会下雨吗？",
        "好的，那我需要带伞吗？",
        "下周呢？"
    ]

    chat_history_ids = None

    print("\n" + "="*20 + " 开始多轮对话 " + "="*20)
    for i, user_input in enumerate(user_inputs, 1):
        response, chat_history_ids = generate_response_multiturn(
            user_input,
            chat_history_ids=chat_history_ids,
            max_context_length=256,
            max_new_tokens=50
        )
        print(f"\n[第 {i} 轮对话]")
        print(f"用户: {user_input}")
        print(f"模型: {response}")
        print("-" * 50)