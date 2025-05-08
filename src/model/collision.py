import jieba
import random
import string
import pickle
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ciphermind import CipherMindModel
from transformers import AutoModel
from collections import Counter

def random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

# 以单个字符为单位比较字符串相似度
def cosine_sim_char(str1, str2):
    # 统计字符频率
    count1 = Counter(str1)
    count2 = Counter(str2)
    
    # 合并所有字符作为公共维度
    all_chars = set(count1.keys()).union(set(count2.keys()))
    
    # 构建向量
    vec1 = [count1.get(char, 0) for char in all_chars]
    vec2 = [count2.get(char, 0) for char in all_chars]
    
    # 计算点积和模
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm1 = math.sqrt(sum(v ** 2 for v in vec1))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2))
    
    # 避免除以零
    if norm1 * norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# 以单词/词语为单位比较字符串相似度
def cosine_sim(text1, text2):
    """
    计算两个文本的余弦相似度
    """
    # 中文分词
    words1 = list(jieba.cut(text1))
    words2 = list(jieba.cut(text2))
    # 合并词表
    vocab = list(set(words1 + words2))
    # 生成词频向量
    vec1 = [words1.count(word) for word in vocab]
    vec2 = [words2.count(word) for word in vocab]
    # 计算相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

def collision_test(sender, attacker, max_len=100, sample_per_length=10):
    layer_num = len(model.model.layers)
    score_map = {}
    # 遍历[0, max_len)中的长度，每个长度进行sample_per_length次测试
    # 得到各个长度模型的碰撞相似度
    for length in tqdm(range(max_len)):
        score_map[length] = 0
        for _ in range(sample_per_length):
            text = random_string(length)
            input_ids = sender.init_input_ids(text)

            idx = 0
            output = ""
            while True:
                hidden_states, state, input_ids = sender.sender_step(input_ids, idx)
                if state == -2:
                    # 得到了多余的token
                    continue
                idx += 1

                if state < 0 and state != -2:
                    # print("Receive:", output)
                    if state == -1:
                        score_map[length] += cosine_sim_char(text, output) / sample_per_length
                    else:
                        output = ""
                    attacker.receiver_reset()
                    break

                # TODO: simulate the attacker diff
                out_layer = random.randint(0, layer_num - 1)
                output = attacker.receiver_step_for_experiment(hidden_states, out_layer)
    return score_map

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sender = CipherMindModel(model, tokenizer)
    attacker = CipherMindModel(model, tokenizer)
    score_map = collision_test(sender, attacker)

    print(score_map)

    with open('../../data/res/collision/collision_char.pkl', 'wb') as file:
        pickle.dump(score_map, file)