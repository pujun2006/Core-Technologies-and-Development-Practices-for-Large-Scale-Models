import gc
import pandas as pd

# 【修正方案 2】：使用 joblib 显式管理并行后端，避免进程间 memoryview 锁死
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# 1. 构建模拟数据集
data = pd.DataFrame({
    "text": [
        "这是分类任务的第一个样本。",
        "文本分类依赖于数据的高质量。",
        "机器学习模型需要不断调优。",
        "分类模型能够自动学习特征。",
        "调整超参数可以提升模型效果。",
        "自然语言处理需要合理的预处理和特征提取。",
        "管道技术可以防止交叉验证中的数据泄露。",
        "随机森林在很多文本分类任务上表现良好。",
        "超参数网格搜索帮助我们找到最优的模型配置。",
        "准确率是评估分类模型性能的常用指标之一。",
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
})

# 2. 划分训练集和测试集
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["label"]
)

# 3. 定义文本特征提取和分类器的 Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(random_state=42)),
])

# 4. 定义超参数搜索空间
param_grid = {
    "tfidf__max_df": [0.9, 1.0],
    "tfidf__min_df": [1],
    "clf__n_estimators": [10, 50],
    "clf__max_depth": [5, 10, None],
    "clf__min_samples_split": [2, 5],
}

# 5. 使用 GridSearchCV 进行超参数调优
# 【关键修正 1】：推荐设置 n_jobs=1 (或搭配线程池 backend)，防止多进程并发序列化导致 memoryview 锁死
grid_search = GridSearchCV(
    pipeline, param_grid, cv=2, scoring="accuracy", n_jobs=1, verbose=1
)

# 【关键修正 2】：如果你仍希望使用多核并行（例如 2 核），可以用 threading 后端包裹 fit：
# with parallel_backend('threading', n_jobs=-1):
#     grid_search.fit(train_data["text"], train_data["label"])

grid_search.fit(train_data["text"], train_data["label"])

# 6. 输出最佳超参数组合与结果
print("\n--- 调优完成 ---")
print("最佳超参数组合:", grid_search.best_params_)
print(f"交叉验证最佳平均准确率: {grid_search.best_score_:.4f}")

# 7. 测试集预测与评估
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(test_data["text"])
print(
    f"测试集准确率: {accuracy_score(test_data['label'], test_predictions):.4f}"
)

# 【关键修正 3】：在程序主线程退出前手动触发垃圾回收，主动销毁并解绑内存缓冲区，防止报 Exception ignored
gc.collect()