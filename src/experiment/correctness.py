import torch
import random
import string
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_string(length):
    """生成指定长度的随机字母数字组合字符串

    Args:
        length (int): 需要生成的字符串长度

    Returns:
        str: 由大小写字母和数字组成的随机字符串
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

# 比较output中是否按顺序包含了input
def compare(input, output):
    """验证输入字符串是否为输出字符串的子序列

    Args:
        input (str): 需要验证的原始输入字符串
        output (str): 模型生成的输出字符串

    Returns:
        bool: 如果input是output的子序列返回True，否则返回False
    """
    iid = 0
    oid = 0
    while iid < len(input) and oid < len(output):
        if input[iid] == output[oid]:
            iid += 1
        oid += 1
    if iid == len(input):
        return True
    else:
        return False

# 模型按照提示复读output
def generate(model, tokenizer, text):
    """调用语言模型生成重复输入文本的响应

    Args:
        model (AutoModelForCausalLM): 加载好的语言模型
        tokenizer (AutoTokenizer): 文本分词器
        text (str): 需要重复的原始文本

    Returns:
        str: 模型生成的响应文本（去除模板内容后的纯文本）
    """
    messages = [
        {"role": "system", "content": "You are a repeater"},
        {"role": "user", "content": f"Repeat in the same case, '{text}'"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=500,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    input_size = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    response = tokenizer.decode(output[0], skip_special_tokens=True)[input_size:]
    return response

def matching_experiment(model, tokenizer, max_len=64, sample_per_length=100):
    """执行模型重复能力的批量测试实验

    Args:
        model (AutoModelForCausalLM): 待测试的语言模型
        tokenizer (AutoTokenizer): 文本分词器
        max_len (int, optional): 测试的最大文本长度，默认100
        sample_per_length (int, optional): 每个长度测试样本数，默认20

    Returns:
        dict: 包含各长度正确率的字典，格式为 {长度: 正确样本数}
    """
    correct_map = {}
    # 遍历[0, max_len)中的长度，每个长度进行sample_per_length次测试
    # 得到各个长度模型的成功传输率
    for length in tqdm(range(max_len)):
        if length == 0:
            correct_map[length] = 100
            continue
        correct_map[length] = 0
        # for _ in range(sample_per_length):
            # text = random_string(length)# 随机无意义字符串试验方法
        with open(f'../model/corpora/length_{length}.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                text = line.strip()# 随机有意义文本试验方法
                output = generate(model, tokenizer, text)
                if compare(text, output):
                    correct_map[length] += 1
    return correct_map

if __name__ == "__main__":
    # base model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.backends.cudnn.benchmark = True

    base_map = matching_experiment(model, tokenizer)
    print("Base Model,", base_map)

    with open('../../data/res/correctness/token/base_token_map.pkl', 'wb') as file:
        pickle.dump(base_map, file)

    # tunned (lora) model
    # tunned_model_name = "../../data/models/tunning_Math500_1000_0"
    # tunned_model = AutoModelForCausalLM.from_pretrained(tunned_model_name).to(device)

    
    # tunned_map = matching_experiment(tunned_model, tokenizer)
    # print("tunned Model,", tunned_map)
    # ver = 0
    # while os.path.exists(f"../../data/res/correctness/token/tunned1000_Math_map_token_v{ver}.pkl"):
    #     ver += 1
    # with open(f'../../data/res/correctness/token/tunned1000_Math_map_token_v{ver}.pkl', 'wb') as file:
    #     pickle.dump(tunned_map, file)
    # print(ver)