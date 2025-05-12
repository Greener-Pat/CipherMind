from accelerate.utils import add_model_config_to_megatron_parser
import jieba
import random
import string
import pickle
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..model.ciphermind import CipherMindModel
from transformers import AutoModel
from collections import Counter
import os
def random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

# 以单个字符为单位比较字符串相似度
def cosine_sim_char(str1, str2):
    """
    计算两个字符串在字符级别的余弦相似度

    Args:
        str1 (str): 第一个输入字符串
        str2 (str): 第二个输入字符串

    Returns:
        float: 余弦相似度得分，范围[0,1]
    """
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
    计算两个文本在词频级别的余弦相似度（支持中文分词）

    Args:
        text1 (str): 第一个文本内容
        text2 (str): 第二个文本内容

    Returns:
        float: 基于词频向量的余弦相似度，范围[0,1]
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

def collision_test(sender, attacker, max_len = 100, sample_per_length = 20):    
    """
    执行字符级碰撞测试实验，统计不同长度字符串的恢复准确率

    Args:
        sender (CipherMindModel): 发送方模型实例
        attacker (CipherMindModel): 攻击方模型实例
        max_len (int, optional): 最大测试字符串长度，默认100
        sample_per_length (int, optional): 每个长度采样次数，默认20

    Returns:
        dict: 包含各长度平均相似度的字典，格式为 {长度: 相似度总分}
    """
    layer_num = len(sender.layers)
    score_map = {}
    fail_map = {}
    # 遍历[0, max_len)中的长度，每个长度进行sample_per_length次测试
    # 得到各个长度模型的碰撞相似度
    for length in tqdm(range(max_len)):
        score_map[length] = 0
        fail_map[length] = 0
        success_count = 0
        defeat_count = 0
        while(success_count < sample_per_length):
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
                        success_count += 1
                    else:
                        output = ""
                        fail_map[length] += 1
                    attacker.receiver_reset()

                    break

                # TODO: simulate the attacker diff
                out_layer = random.randint(0, layer_num - 1)
                output = attacker.receiver_step_for_experiment(hidden_states, out_layer)
    return score_map

if __name__ == "__main__":
    base_name = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_name = "../../data/models/lora_model"
    tokenizer = AutoTokenizer.from_pretrained(base_name)

    lora_model = AutoModelForCausalLM.from_pretrained("../../data/models/lora_model")
    sender = CipherMindModel(lora_model, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(base_name)
    attacker = CipherMindModel(base_model, tokenizer)

    score_map = collision_test(sender, attacker)
    print(score_map)
    base_path = '../../data/res/collision/base_base/collision_char'

    version = 1
    while os.path.exists(f"{base_path}_v{version}.pkl"):
        version += 1
    with open(f"{base_path}_v{version}.pkl", 'wb') as file:
        pickle.dump(score_map, file)
    with open(f"{base_path}_fail_map_v{version}.pkl", 'wb') as file:
        pickle.dump(score_map, file)
    
    