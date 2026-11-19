import re
import jieba
from transformers import AutoTokenizer

# 1. 示例文本
texts = [
    "自然语言处理的分词方法可以提升模型的理解能力。",
    "词级分词与子词分词适用于不同的NLP任务。",
    "BERT模型常用子词分词以处理大量词汇。",
]


# 2. 正确的中文词级分词实现（基于语义词）
def word_level_tokenization(text):
    """使用 jieba 进行真正的中文词级分词（词语粒度）"""
    # 过滤掉标点符号，仅保留词语
    text_clean = re.sub(r"[^\w\s]", "", text)
    tokens = jieba.lcut(text_clean)
    return tokens


# 3. Hugging Face 子词分词实现（基于 WordPiece/字粒度）
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def subword_level_tokenization(text, tokenizer):
    """使用 BERT Tokenizer 进行子词分词"""
    subwords = tokenizer.tokenize(text)
    return subwords


# 4. 执行并对比两种分词结果
for text in texts:
    word_tokens = word_level_tokenization(text)
    subword_tokens = subword_level_tokenization(text, tokenizer)

    print(f"原始文本: {text}")
    print(f"词级分词 (Word-level):    {word_tokens}")
    print(f"子词分词 (Subword-level): {subword_tokens}")
    print("-" * 60)