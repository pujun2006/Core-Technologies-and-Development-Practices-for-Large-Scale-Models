from collections import Counter, defaultdict
from transformers import AutoTokenizer

# 1. 准备具备词频信息（Frequency）的语料词表 (Word Corpus Frequencies)
corpus_vocab = Counter({
    "自然语言": 10,
    "处理": 8,
    "分词": 15,
    "方法": 5,
    "提升": 7,
    "模型": 12,
    "理解": 6,
    "能力": 9,
})

texts = ["自然语言处理的分词方法可以提升模型的理解能力。"]


# ==========================================
# 2. 编写正确的 BPE 算法 (训练 + 编码)
# ==========================================
def train_bpe(corpus_vocab, num_merges=10):
    """根据语料词频构建 BPE 合并规则 (Merges) 与 最终子词词表 (Subword Vocab)"""
    # 拆分为字符列表并在末尾加上 </w>
    bpe_corpus = {
        tuple(list(word) + ["</w>"]): freq
        for word, freq in corpus_vocab.items()
    }
    merges = []  # 记录按优先级排序的合并规则

    for i in range(num_merges):
        pairs = defaultdict(int)
        for tokens, freq in bpe_corpus.items():
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j + 1])] += freq

        if not pairs:
            break

        # 找到出现总频次最高的相邻字符对
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # 更新语料表中的 Token 结构
        new_bpe_corpus = {}
        for tokens, freq in bpe_corpus.items():
            new_tokens = []
            j = 0
            while j < len(tokens):
                if (
                    j < len(tokens) - 1
                    and (tokens[j], tokens[j + 1]) == best_pair
                ):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            new_bpe_corpus[tuple(new_tokens)] = freq
        bpe_corpus = new_bpe_corpus

    # 提取学习到的所有子词集 (Subword Vocabulary)
    subword_vocab = set()
    for tokens in bpe_corpus.keys():
        for token in tokens:
            subword_vocab.add(token)

    return merges, subword_vocab


# 执行 BPE 训练
merges_rules, bpe_subwords = train_bpe(corpus_vocab, num_merges=10)


def bpe_encode(text, subword_vocab):
    """使用训练好的子词词表对新文本进行最长匹配 (Longest Match) 分词"""
    tokens = list(text)
    encoded_text = []
    i = 0
    while i < len(tokens):
        matched = False
        # 从长到短进行最长贪心匹配
        for j in range(len(tokens), i, -1):
            subword = "".join(tokens[i:j])
            if subword in subword_vocab or subword + "</w>" in subword_vocab:
                encoded_text.append(subword)
                i = j
                matched = True
                break
        if not matched:
            encoded_text.append(tokens[i])
            i += 1
    return encoded_text


# ==========================================
# 3. Hugging Face WordPiece 分词
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def wordpiece_tokenization(text, tokenizer):
    return tokenizer.tokenize(text)


# ==========================================
# 4. 执行并对比结果
# ==========================================
print("=== 1. 手写 BPE 分词算法 ===")
for text in texts:
    bpe_tokens = bpe_encode(text, bpe_subwords)
    print(f"原始文本: {text}")
    print(f"BPE 编码结果: {bpe_tokens}\n")

print("=== 2. Hugging Face WordPiece (bert-base-chinese) ===")
for text in texts:
    wp_tokens = wordpiece_tokenization(text, tokenizer)
    print(f"原始文本: {text}")
    print(f"WordPiece 结果: {wp_tokens}\n")