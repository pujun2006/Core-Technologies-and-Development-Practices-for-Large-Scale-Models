import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# 1. 定义网络结构（修正了 Conv2d 后的全连接层维度：以 28x28 输入为例）
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # 26x26 -> 24x24
        self.fc1 = nn.Linear(32 * 24 * 24, 128)  # 修正输入维度为 18432
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 2. 统计模型稀疏度的标准函数
def print_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            # 兼容 PyTorch 剪枝钩子生成的 weight
            w = module.weight
            num_zeros = torch.sum(w == 0).item()
            num_elements = w.nelement()
            sparsity = 100.0 * num_zeros / num_elements
            print(f"层 [{name}] 稀疏度: {sparsity:.2f}%")
            total_zeros += num_zeros
            total_elements += num_elements

    if total_elements > 0:
        print(
            f"--> 全模型总稀疏度: {100. * total_zeros / total_elements:.2f}%\n"
        )


# 3. 非结构化剪枝 (Unstructured Pruning)
def apply_unstructured_pruning(model, amount=0.5):
    prune.random_unstructured(model.fc1, name="weight", amount=amount)
    prune.random_unstructured(model.fc2, name="weight", amount=amount)
    return model


# 4. 结构化剪枝 (Structured Pruning)
def apply_structured_pruning(model, amount=0.3):
    # 对 conv1/conv2 按 Channel (dim=0) 进行 L2 范数剪枝
    prune.ln_structured(model.conv1, name="weight", amount=amount, n=2, dim=0)
    prune.ln_structured(model.conv2, name="weight", amount=amount, n=2, dim=0)
    return model


# 5. 自定义 L1 阈值剪枝（正确处理 mask 与数据覆盖）
def apply_custom_magnitude_pruning(model, threshold_quantile=0.2):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 如果之前调用过 official prune，先移除 Hook 固化权重
            if prune.is_pruned(module):
                prune.remove(module, "weight")

            with torch.no_grad():
                importance_scores = torch.abs(module.weight)
                # 计算分位数阈值
                threshold_value = torch.quantile(
                    importance_scores, threshold_quantile
                )
                mask = importance_scores > threshold_value
                # 显式覆盖 weight 的数据
                module.weight.mul_(mask.float())
    return model


# ==========================================
# 主运行程序
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)

    print("--- 1. 应用非结构化剪枝 ---")
    model = apply_unstructured_pruning(model, amount=0.5)
    print_sparsity(model)

    print("--- 2. 应用结构化剪枝 ---")
    model = apply_structured_pruning(model, amount=0.3)
    print_sparsity(model)

    print("--- 3. 固化官方剪枝并应用自定义重要性剪枝 ---")
    model = apply_custom_magnitude_pruning(model, threshold_quantile=0.2)
    print_sparsity(model)