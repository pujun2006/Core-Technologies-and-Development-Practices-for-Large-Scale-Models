from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ==============================================================================
# 1. 构建模拟数据集 (文本与分类标签)
# ==============================================================================
raw_data = pd.DataFrame({
    "text": [
        "这是第一个文本。包含标点和一些不需要的字符！",
        "第二个文本：数据清理非常重要！",
        "机器学习是未来，深度学习非常强大。    ",
        "文本预处理可以大幅改善模型性能。",
        "分类任务需要高质量的数据集与恰当的预处理。",
        "数据泄露是机器学习模型评估中最常见的问题之一！",
        "通过正确划分训练集与测试集，可以避免过拟合问题。",
        "自然语言处理包含分词、文本向量化以及模型训练。",
        "Python 和 Scikit-learn 是数据科学家的必备工具。",
        "正确的交叉验证能够科学评估模型泛化性能。",
    ],
    "label": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # 类别比例不平衡: 1较多，0较少
})


# ==============================================================================
# 2. 文本清洗函数 (Text Cleaning)
# ==============================================================================
def clean_text(text: str) -> str:
    """清洗文本：去除非文本标点符号、规范化空格并转小写"""
    # 1. 去除特殊字符与标点（保留汉字 \u4e00-\u9fa5、英文字母、数字和空格）
    text = re.sub(r"[^\w\s]", "", text)
    # 2. 将多个连续空格替换为单个空格，并去掉首尾空格
    text = re.sub(r"\s+", " ", text).strip()
    # 3. 转小写（主要针对文本中的英文字符）
    text = text.lower()
    return text


print("--- 步骤 1: 文本清洗 ---")
raw_data["cleaned_text"] = raw_data["text"].apply(clean_text)
print("清洗后数据示例:\n", raw_data[["text", "cleaned_text"]].head(3))


# ==============================================================================
# 3. 严谨的训练集 / 测试集划分 (Train-Test Split)
# 【核心原则 1】：必须在任何重采样/特征提取之前进行划分！
# ==============================================================================
print("\n--- 步骤 2: 划分训练集与测试集 ---")
X = raw_data["cleaned_text"]
y = raw_data["label"]

X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,  # 按标签比例分层抽样，确保训练集和测试集类别比例一致
)

print("原始训练集标签分布:", Counter(y_train_raw))
print("测试集标签分布 (保持真实分布):", Counter(y_test))


# ==============================================================================
# 4. 训练集数据平衡 (Over-sampling Training Data Only)
# 【核心原则 2】：仅对训练集做过采样，绝对不能触碰测试集！
# ==============================================================================
print("\n--- 步骤 3: 训练集过采样平衡 ---")
train_df = pd.concat(
    [X_train_raw, y_train_raw], axis=1
)  # 组装当前训练集 DataFrame

majority_class = train_df[train_df["label"] == 1]
minority_class = train_df[train_df["label"] == 0]

# 对训练集的少数类进行过采样
minority_upsampled = resample(
    minority_class,
    replace=True,  # 允许重复采样
    n_samples=len(majority_class),  # 调整至与多数类数量相同
    random_state=42,
)

# 合并生成最终的平衡训练集
train_df_balanced = pd.concat([majority_class, minority_upsampled])

X_train_balanced = train_df_balanced["cleaned_text"]
y_train_balanced = train_df_balanced["label"]

print("平衡后训练集标签分布:", Counter(y_train_balanced))


# ==============================================================================
# 5. 特征提取 - TF-IDF 向量化 (TF-IDF Vectorization)
# 【核心原则 3】：fit 只能作用于训练集，对测试集只能执行 transform！
# ==============================================================================
print("\n--- 步骤 4: 文本向量化 (TF-IDF) ---")

# 在实际中文生产环境中，建议传入 tokenizer 参数（如 jieba.cut）
vectorizer = TfidfVectorizer()

# 在平衡后的训练集上拟合词表并生成特征矩阵
X_train_tfidf = vectorizer.fit_transform(X_train_balanced)

# 仅使用训练集学到的词表转换测试集（防止数据泄露）
X_test_tfidf = vectorizer.transform(X_test_raw)

print(f"训练集特征矩阵形状: {X_train_tfidf.shape}")
print(f"测试集特征矩阵形状:   {X_test_tfidf.shape}")


# ==============================================================================
# 6. 模型训练与评估 (Model Training & Evaluation)
# ==============================================================================
print("\n--- 步骤 5: 模型训练与评估 ---")

# 初始化分类器（逻辑回归）
clf = LogisticRegression(random_state=42)

# 使用处理好的训练数据拟合模型
clf.fit(X_train_tfidf, y_train_balanced)

# 在独立的测试集上进行预测
y_pred = clf.predict(X_test_tfidf)

# 输出评估报告
print("\n分类性能评估报告 (Classification Report):")
print(
    classification_report(
        y_test, y_pred, target_names=["Class 0", "Class 1"], zero_division=0
    )
)

print("混淆矩阵 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))