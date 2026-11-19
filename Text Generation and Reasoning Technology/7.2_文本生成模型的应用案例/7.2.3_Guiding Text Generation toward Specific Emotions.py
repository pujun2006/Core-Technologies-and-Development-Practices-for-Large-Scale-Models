import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载中文 GPT-2 模型与分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置防御性属性
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token_id = tokenizer.eos_token_id


# ==========================================
# 规范的情绪文本生成函数（基于 Prompt 文本引导）
# ==========================================
def generate_emotion_text_correct(
    prompt, emotion_desc="", max_new_tokens=50, temperature=0.7, top_p=0.9
):
    """
    prompt: 核心上下文/开头
    emotion_desc: 情绪风格的自然语言描述（如 "极其开心、积极向上" 或 "悲伤、消极沮丧"）
    """
    # 通过自然的文本 Prompt 引导模型的情绪走向，而不是使用无效的伪 Tag
    if emotion_desc:
        full_prompt = f"【请用{emotion_desc}的情绪接写】\n{prompt}"
    else:
        full_prompt = prompt

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=50,
            temperature=temperature,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 仅提取模型新生成的内容
    response_ids = output[:, input_ids.shape[-1] :]
    generated_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return generated_text


# ==========================================
# 测试生成
# ==========================================
base_prompt = "今天的天气真好，"

print("--- 1. 正面/积极情绪生成 ---")
pos_text = generate_emotion_text_correct(
    prompt=base_prompt,
    emotion_desc="极其激动、开心和阳光",
    temperature=0.8,  # 稍微提高温度增加表达丰富度
)
print("生成的正面文本:", base_prompt + pos_text)

print("\n--- 2. 负面/悲伤情绪生成 ---")
neg_text = generate_emotion_text_correct(
    prompt=base_prompt,
    emotion_desc="十分沮丧、悲伤和消极",
    temperature=0.8,
)
print("生成的负面文本:", base_prompt + neg_text)