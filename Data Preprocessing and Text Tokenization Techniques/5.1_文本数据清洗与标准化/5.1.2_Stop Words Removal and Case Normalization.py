import re
import jieba

# 1. 停用词表
stopwords = {"的", "和", "在", "是", "了", "也", "有", "与", "对", "就"}

# 2. 示例文本数据
texts = [
    "在自然语言处理中，停用词的去除和大小写规范化是重要的一步。",
    "停用词的存在会增加数据维度，减少模型的计算效率。",
    "大小写不统一可能会导致同一词被视为不同词项。",
]


def clean_text_remove_stopwords(text):
    """去除中文文本中的标点符号、停用词，并转换英文字母为小写"""
    # 1. 去除标点符号并转为小写
    text = re.sub(r"[^\w\s]", "", text).lower()

    # 2. 正确的中文分词（使用 jieba）
    words = jieba.lcut(text)

    # 3. 过滤停用词与空白字符
    words = [
        word.strip()
        for word in words
        if word.strip() and word not in stopwords
    ]

    # 4. 重新拼接文本
    processed_text = " ".join(words)
    return processed_text


# 3. 对文本数据进行处理
processed_texts = [clean_text_remove_stopwords(text) for text in texts]

# 4. 输出处理结果
for i, text in enumerate(processed_texts):
    print(f"原始文本:\n{texts[i]}")
    print(f"处理后文本:\n{text}\n")