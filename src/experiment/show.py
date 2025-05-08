import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_mmlu():
    with open('../../data/res/mmlu/base_mmlu.pkl', 'rb') as file:
        base_acc = pickle.load(file)
    with open('../../data/res/mmlu/lora_mmlu.pkl', 'rb') as file:
        lora_acc = pickle.load(file)

    # 提取键和值
    categories = list(base_acc.keys())
    values1 = list(base_acc.values())
    values2 = list(lora_acc.values())

    # 设置柱状图参数
    bar_width = 0.35
    x = np.arange(len(categories))

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width/2, values1, width=bar_width, label='base model')
    plt.bar(x + bar_width/2, values2, width=bar_width, label='tunned model')

    # 添加标签和标题
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.xticks(x, categories)
    plt.legend()
    plt.title('Comparison of MMLU Scores')
    plt.show()

def show_correctness():
    # with open('../../data/res/correctness/base_map.pkl', 'rb') as file:
    #     base_map = pickle.load(file)

    with open('../../data/res/correctness/lora_map.pkl', 'rb') as file:
        lora_map = pickle.load(file)

    # base_list = list(base_map.values())
    lora_list = torch.tensor(list(lora_map.values())) / 20

    plt.figure()
    plt.ylabel('Succeed Send Rate')
    plt.xlabel('Send Length')
    # plt.plot(base_list)
    plt.plot(lora_list)
    plt.legend()
    plt.title('Transmission Capability')
    plt.show()

def show_collision():
    with open('../../data/res/collision/collision.pkl', 'rb') as file:
        collision_dict = pickle.load(file)

    print(collision_dict)

    # base_list = list(base_map.values())
    collision_list = list(collision_dict.values())

    plt.figure()
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Send Length')
    plt.plot(collision_list)
    plt.legend()
    plt.title('Collision Test')
    plt.show()

if __name__ == "__main__":
    show_collision()
    show_correctness()
    show_mmlu()