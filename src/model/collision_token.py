from accelerate.utils import add_model_config_to_megatron_parser
from pandas.core.indexing import length_of_indexer
import torch
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
    all_chars = set(count1) | set(count2)
    
    # 向量化计算
    vec1 = np.array([count1.get(c, 0) for c in all_chars], dtype=np.float32)
    vec2 = np.array([count2.get(c, 0) for c in all_chars], dtype=np.float32)
    
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / norm if norm != 0 else 0.0

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

def collision_test(sender, attacker, max_drop = 20, max_len = 64, sample_per_length = 100, IsTransparent = False):    
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
    for length in tqdm(range(0,max_len+1)):
        score_map[length] = 0
        fail_map[length] = 0
        success = 0
        while(success + fail_map[length] < sample_per_length):
            with open(f'corpora/length_{length}.txt') as f:
                lines = f.readlines()
            text = random.choice(lines).strip()
            input_ids = sender.init_input_ids(text)
            drop = 0
            idx = 0
            output = ""
            while True:
                hidden_states, state, input_ids = sender.sender_step(input_ids, idx)
                if state == -2:
                    drop += 1
                    if drop > max_drop:
                        output = ""
                        print("fail to send(-2)")
                        fail_map[length] += 1
                        drop = 0
                        break
                    # 得到了多余的token
                    continue
                idx += 1

                if state < 0 and state != -2:
                    # print("Receive:", output)
                    if state == -1:
                        score_map[length] += cosine_sim_char(text, output)
                        success += 1
                    else:
                        output = ""
                        fail_map[length] += 1
                    attacker.receiver_reset()

                    break

                # TODO: simulate the attacker diff
                if IsTransparent:
                    out_layer = sender.middle_layer
                else:
                    out_layer = random.randint(0, layer_num - 1)
                output = attacker.receiver_step_for_experiment(hidden_states, out_layer)
        score_map[length] /= success   
    return score_map, fail_map
if __name__ == "__main__":
    base_name = "Qwen/Qwen2.5-0.5B-Instruct"
    sender_step = 1000
    tunned_name = f"../../data/models/tunning_Math500_{sender_step}_0"
    # 预加载模型到内存
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_name).to('cuda')  # 使用GPU加速
    
    # 复用基础模型
    attacker = CipherMindModel(base_model, tokenizer)
    sender = CipherMindModel(base_model, tokenizer)
    tunned_model = AutoModelForCausalLM.from_pretrained(tunned_name).to('cuda')
    sender = CipherMindModel(tunned_model, tokenizer)

    score_map,fail_map = collision_test(sender, attacker)
    print(score_map)
    base_path = f'../../data/res/collision/tune_base/collision_char_Math500_{sender_step}_0'

    
    attacker_step = 0
    version = 1
    
    while os.path.exists(f"{base_path}_v{version}.pkl"):
        version += 1
    with open(f"{base_path}_v{version}.pkl", 'wb') as file:
        pickle.dump(score_map, file)
    with open(f"{base_path}_fail_map_v{version}.pkl", 'wb') as file:
        pickle.dump(fail_map, file)    
    