import torch
from torch.optim import AdamW  # ✅ 使用 PyTorch 原生的 AdamW 替代 transformers.AdamW
from transformers import AutoModel, BertConfig, BertModel

# ----------------------------------------------------
# 1. 自定义配置从头初始化模型 (Scratch Training 场景)
# ----------------------------------------------------
config = BertConfig(
    hidden_size=512,  # 隐藏层大小
    num_attention_heads=8,  # 注意力头数
    num_hidden_layers=6,  # Transformer层数
)
print("自定义的模型配置:", config)

model_scratch = BertModel(config)
print("基于自定义配置初始化的模型加载成功！")


# ----------------------------------------------------
# 2. 加载预训练模型并覆盖 Dropout 参数
# ----------------------------------------------------
model_name = "bert-base-uncased"

# ✅ 最佳实践：在加载时传入 Dropout 参数，确保模型内部的 Dropout 层立刻生效
model = AutoModel.from_pretrained(
    model_name, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3
)
print("调整后的隐藏层 Dropout 概率:", model.config.hidden_dropout_prob)


# ----------------------------------------------------
# 3. 冻结 BERT 前 4 层 Transformer Encoder 的参数
# ----------------------------------------------------
for idx, layer in enumerate(model.encoder.layer):
    if idx < 4:
        for param in layer.parameters():
            param.requires_grad = False

# 计算总参数与可训练参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"前 4 层已冻结 | 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}"
)


# ----------------------------------------------------
# 4. 配置优化器（仅对 requires_grad=True 的参数进行优化）
# ----------------------------------------------------
# 最佳实践：只将未冻结的参数传给优化器
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01,
)
print("自定义优化器配置成功，学习率:", optimizer.defaults["lr"])


# ----------------------------------------------------
# 5. 保存与重新加载模型
# ----------------------------------------------------
# 注意：model.save_pretrained 会自动把 model.config 一并保存，无需重复调用 model.config.save_pretrained
model.save_pretrained("./customized_model")
print("调整后的模型和配置已保存至 ./customized_model")

custom_model = AutoModel.from_pretrained("./customized_model")
print("自定义配置的模型加载成功！")