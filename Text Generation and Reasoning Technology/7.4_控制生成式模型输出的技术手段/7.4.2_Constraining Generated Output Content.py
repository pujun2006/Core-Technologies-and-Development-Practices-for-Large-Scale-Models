import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# 1. 设置设备与加载适配中文的模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 2. 自定义停止条件（支持任意字符串 stop_token）
# ==========================================
class StringStoppingCriteria(StoppingCriteria):

    def __init__(self, stop_string, tokenizer, prompt_length):
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # 只提取新生成的 Token 进行解码判断
        generated_text = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )
        return self.stop_string in generated_text


# ==========================================
# 3. 规范的受限生成函数
# ==========================================
def generate_text_with_constraints(
    prompt,
    max_new_tokens=80,
    temperature=0.7,
    bad_words=None,
    stop_token=None,
):
    """使用 Transformers 官方原生接口限制禁用词与停止标记。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    # 1. 构建 bad_words_ids（Transformers 官方原生支持）
    bad_words_ids = []
    if bad_words:
        for word in bad_words:
            ids = tokenizer(word, add_special_tokens=False).input_ids
            if ids:
                bad_words_ids.append(ids)

    # 2. 构建停止条件
    stopping_criteria = StoppingCriteriaList()
    if stop_token:
        stopping_criteria.append(
            StringStoppingCriteria(stop_token, tokenizer, prompt_len)
        )

    # 3. 生成文本
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            bad_words_ids=bad_words_ids if bad_words_ids else None,  # 传入黑名单
            stopping_criteria=stopping_criteria,  # 传入停止条件
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 精准切片与解构
    generated_ids = output[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 如果存在 stop_token，截取其第一次出现之前的内容
    if stop_token and stop_token in generated_text:
        generated_text = generated_text.split(stop_token)[0] + stop_token

    return prompt + generated_text


# ==========================================
# 4. 主题引导生成函数
# ==========================================
def generate_with_topic(prompt, topic_words, max_new_tokens=80, temperature=0.7):
    # 使用 Prompt 拼接引导主题（System Prompt / Pre-prompt 范式）
    topic_prompt = f"【主题：{'、'.join(topic_words)}】{prompt}"
    return generate_text_with_constraints(
        topic_prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )


# ==========================================
# 5. 测试运行
# ==========================================
if __name__ == "__main__":
    prompt = "在未来的科技世界中，"
    bad_words = ["战争", "冲突"]  # 不允许生成的词汇
    stop_token = "。"  # 遇到第一个句号自动停止

    print("--- 测试黑名单与强制停止符 ---")
    restricted_text = generate_text_with_constraints(
        prompt, bad_words=bad_words, stop_token=stop_token
    )
    print("生成的文本:\n", restricted_text)
    print("-" * 50)

    print("\n--- 测试主题引导生成 ---")
    topic_words = ["智能家居", "人工智能"]
    topic_restricted_text = generate_with_topic(
        "未来的生活将如何改变？", topic_words
    )
    print("主题生成文本:\n", topic_restricted_text)