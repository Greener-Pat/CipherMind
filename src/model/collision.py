import jieba
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ciphermind import CipherMindModel
from transformers import AutoModel

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

def get_data():
    text = []
    with open("../../data/text/small_news.txt", "r", encoding="utf-8") as f:
        for line in f:
            text.append(line.strip())
    return text

def collision_test(model):
    # dataset
    dataset = get_data()
    
    # models
    # TODO: use the model send in

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tunned_model = AutoModel.from_pretrained("../../data/models/tunning0")
    model.model = tunned_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # base_model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    sender = CipherMindModel(model, tokenizer)
    attacker = CipherMindModel(model, tokenizer)
    
    layer_num = len(model.model.layers)

    score = 0
    total_length = 0
    count = 0
    for text in dataset:
        if text.find("？") != -1 or text.find("！") != -1:#为了减少不合格的语料？
            continue
        length = len(text)
        new_text = text.replace("_!_", " ").strip()
        new_text = "star"
        print("Sending:", new_text)

        input_ids = sender.init_input_ids(new_text)
        idx = 0
        output = ""
        while True:
            hidden_states, state, input_ids = sender.sender_step(input_ids, idx)
            # if state == -2:#得到了多余的token
            #     continue
            idx += 1

            print(state)
            if state < 0 and state != -2:
                print("Receive:", output)
                if state == -1:
                    total_length += length
                    score += cosine_sim(new_text, output) * length
                    count += 1
                else:
                    output = ""
                attacker.receiver_reset()
                break

            # TODO: simulate the attacker diff
            # out_layer = random.randint(0, layer_num - 1)
            # output = attacker.receiver_step_for_experiment(hidden_states, out_layer)
            output = attacker.receiver_step(hidden_states)
            print(output)
        if count > 10:
            break
    score /= total_length
    print("Collision score: ", score)

collision_test(None)