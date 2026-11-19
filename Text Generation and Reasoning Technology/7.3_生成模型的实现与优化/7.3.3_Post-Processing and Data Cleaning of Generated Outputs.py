import re


# ==========================================
# 1. 针对中文优化后的文本清洗函数
# ==========================================
def clean_generated_text_chinese(text):
    """对生成的中文文本进行清洗和规范化。"""
    if not text:
        return ""

    # 1. 将多个连续空格替换为单个空格，并去除首尾空格
    text = re.sub(r"\s+", " ", text).strip()

    # 2. 去除中文连续重复词（如 "科技 科技" -> "科技"，"带来 带来" -> "带来"）
    # 使用 [\u4e00-\u9fa5]+ 匹配连续汉字
    text = re.sub(r"([\u4e00-\u9fa5]+)\s+\1", r"\1", text)

    # 3. 清洗标点符号前的空格（中文标点前后不留多余空格）
    text = re.sub(r"\s+([，。！？；：“”‘’（）,.!?])", r"\1", text)

    # 4. 去除中文标点符号后面的多余空格（英文后保留 1 个空格，中文后不留空格）
    text = re.sub(r"([，。！？；：])\s+", r"\1", text)
    text = re.sub(r"([,.!?])\s*", r"\1 ", text)  # 英文标点统一加一个空格

    # 5. 清除末尾残留的孤立符号或空格
    text = text.strip()

    return text


# ==========================================
# 2. 中文繁简转换与错别字校对函数
# ==========================================
# 提示：需要提前安装 zhconv 库：pip install zhconv
try:
    import zhconv

    def correct_chinese_text(text):
        """将繁体中文转换为简体中文（处理繁体字）"""
        return zhconv.convert(text, "zh-hans")

except ImportError:
    # 未安装 zhconv 时的备选方案
    print("提示：未安装 zhconv 库，将跳过繁简转换。可通过 pip install zhconv 安装。")

    def correct_chinese_text(text):
        return text


# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    # 示例 1：去除多余空格与重复词
    raw_texts = [
        "未来的科技  科技 将会带来  带来新的机遇 和挑战。",
        "智能家居 是未来的发展 方向 , 它将  提高生活质量 。 ",
    ]

    print("--- 1. 中文文本清洗测试 ---")
    for original in raw_texts:
        cleaned = clean_generated_text_chinese(original)
        print("原始文本:", original)
        print("清洗后  :", cleaned)
        print("-" * 50)

    # 示例 2：繁简转换与错字校正
    text_with_typos = "未來的科技將會帶來新地機遇 和挑戰。"
    print("\n--- 2. 繁简转换与综合清洗测试 ---")
    converted_text = correct_chinese_text(text_with_typos)
    final_text = clean_generated_text_chinese(converted_text)

    print("原始文本:", text_with_typos)
    print("转换后  :", final_text)