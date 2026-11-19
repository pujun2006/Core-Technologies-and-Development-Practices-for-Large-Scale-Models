import re

# ==========================================
# 1. 英文文本清洗（保留缩写单引号）
# ==========================================
english_texts = [
    "Hello! This is a sample text with symbols #@! and numbers 1234.",
    "Natural Language Processing (NLP) is exciting!!!",
    "Let's clean up this text: remove symbols & extra spaces.",
]


def clean_english_text(text):
    # 移除 URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # 移除非字母、非数字、非单引号的标点符号（保留 Let's 中的 '）
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)
    # 替换多个空格为一个空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("=== 英文文本清洗结果 ===")
for text in english_texts:
    print(f"清洗前: {text}")
    print(f"清洗后: {clean_english_text(text)}\n")


# ==========================================
# 2. 中文文本清洗（修复邮箱与 URL 正则）
# ==========================================
chinese_texts = [
    "在自然语言处理（NLP）领域，处理特殊字符和标点是非常重要的一步！例如，很多文本中会包含各种符号，如@、#、$、%等，甚至可能包含网址https://example.com，以及电子邮件地址example@example.com。这些符号往往对模型没有帮助，且会增加处理复杂性。此外，重复的标点（如‘！！’）或多余的空格，也会对数据质量产生负面影响。让我们来看看如何清洗这些噪声数据！"
]


def clean_chinese_text(text):
    # 1. 安全移除 URL（只匹配英文字符、数字和 URL 常用符号，遇到中文或中文标点自动停止）
    text = re.sub(r"https?://[a-zA-Z0-9./_?=-]+", "", text)

    # 2. 安全移除 Email
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)

    # 3. 移除非中文、非英文、非数字的特殊符号（中文标点等会被替换为空格）
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", " ", text)

    # 4. 压缩连续空格并去除首尾空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("=== 中文文本清洗结果 ===")
for text in chinese_texts:
    print(f"原始文本:\n{text}\n")
    print(f"清洗后文本:\n{clean_chinese_text(text)}\n")