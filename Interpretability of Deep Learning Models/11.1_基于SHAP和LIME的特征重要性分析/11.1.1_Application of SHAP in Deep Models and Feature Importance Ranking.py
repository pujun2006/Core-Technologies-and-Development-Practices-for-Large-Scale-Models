import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 生成模拟数据集
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 2. 构建深度神经网络模型
class DeepModel(nn.Module):

    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# 初始化模型、损失函数和优化器
model = DeepModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练模型
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 4. 使用 SHAP 计算特征重要性（修正部分）
model.eval()  # 切换至评估模式

# DeepExplainer 背景样本建议抽样（例如前 100~200 条）以加速计算
background = X_train_tensor[:200]
explainer = shap.DeepExplainer(model, background)

# 计算测试集的 SHAP 值 (返回值为 numpy.ndarray)
shap_values = explainer.shap_values(X_test_tensor, check_additivity=False)

# 如果多输出模型返回 3D 数组 [samples, features, outputs]，回归单输出降维至 2D [samples, features]
if len(shap_values.shape) == 3:
    shap_values = shap_values.squeeze(-1)

# 计算每个特征的平均绝对 SHAP 值
feature_importance = np.abs(shap_values).mean(axis=0)

# 5. 排序并显示特征影响力
feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
feature_importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
)
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

print("\n特征影响力排序：")
print(feature_importance_df)

# （可选）绘制 SHAP 蜂窝图
# shap.summary_plot(shap_values, X_test, feature_names=feature_names)