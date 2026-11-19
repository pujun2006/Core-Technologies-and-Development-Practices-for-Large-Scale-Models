from transformers import AutoTokenizer, AutoModel

# 加载预训练的BERT模型和对应的Tokenizer
model_name="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)

print("模型和Tokenizer加载成功！")
# 查看模型的配置信息
print("模型配置:", model.config)

# 定义输入文本
text="Hugging Face's transformers library is powerful for NLP tasks."

# 将文本转化为模型输入格式
inputs=tokenizer(text, return_tensors="pt")
print("输入张量格式:", inputs)

# 模型前向传播，获取输出
outputs=model(**inputs)
print("模型输出:", outputs.last_hidden_state)

# 保存模型和Tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("模型和Tokenizer已保存至 ./saved_model")

# 从本地路径加载模型和Tokenizer
local_model=AutoModel.from_pretrained("./saved_model")
local_tokenizer=AutoTokenizer.from_pretrained("./saved_model")
print("本地模型加载成功！")