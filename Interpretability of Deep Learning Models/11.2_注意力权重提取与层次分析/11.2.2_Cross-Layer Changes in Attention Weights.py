import torch
from transformers import BertTokenizer, BertModel
import seaborn as sns

# ======================================================================
# 关键修复点：在导入 plt 之前指定 Matplotlib 后端，解决 PyCharm backend_interagg 报错
# ======================================================================
import matplotlib
matplotlib.use('TkAgg')  # 如果想弹窗显示图片用 'TkAgg'；如果想静默保存图片用 'Agg'
import matplotlib.pyplot as plt

# 1. 初始化 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# 显式切换为评估模式
model.eval()

# 2. 输入文本示例
sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="pt")

# 将 ID 转换为可读的 Token 列表（含 [CLS] 和 [SEP]）
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 3. 前向传播，禁用梯度计算以节省内存/显存
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # 返回包含 12 层注意力权重的 tuple

# 4. 打印不同层次的注意力权重概览（仅打印形状和局部切片，防止控制台刷屏）
print("分词后的 Tokens:\n", tokens)
print("\n" + "=" * 50)

for layer_idx, layer_attention in enumerate(attentions):
    print(f"Layer {layer_idx+1} Attention Shape: {layer_attention.shape}")
    # 仅展示第 1 个注意力头（Head 0）的前 3x3 局部权重，避免大量数值刷屏
    sample_matrix = layer_attention[0][0][:3, :3].cpu().numpy()
    print(f"Layer {layer_idx+1} (Head 1) 前 3x3 注意力权重切片:\n{sample_matrix}")
    print("=" * 50)


# 5.（可选）绘制最后一层第 1 个头的注意力分布热力图
def plot_attention_heatmap(matrix, tokens, layer_num=12, head_num=1):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
    )
    plt.title(f"Attention Heatmap (Layer {layer_num}, Head {head_num})")
    plt.xlabel("Key Tokens (Attended To)")
    plt.ylabel("Query Tokens (Attending From)")
    plt.tight_layout()
    plt.show()


# 提取第 12 层、第 1 个头的完整矩阵并绘制热力图
last_layer_head1 = attentions[11][0, 0].cpu().numpy()
plot_attention_heatmap(last_layer_head1, tokens)