import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_mmlu():
    """可视化MMLU测试结果对比。
    
    从以下文件加载测试结果：
    - ../../data/res/mmlu/base_mmlu.pkl：原始模型准确率
    - ../../data/res/mmlu/lora_mmlu.pkl：微调后模型准确率
    
    生成双柱状图对比结果，包含：
    - X轴：测试科目分类
    - Y轴：准确率数值
    - 图例：base model / tunned model
    """
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
    """展示消息传输成功率曲线。
    
    从以下文件加载测试结果：
    - ../../data/res/correctness/lora_map.pkl：微调后模型传输数据
    
    生成折线图展示：
    - X轴：消息长度
    - Y轴：成功传输率（数值经过torch.tensor转换及20等分标准化）
    """
    with open('../../data/res/correctness/base_map.pkl', 'rb') as file:
        base_map = pickle.load(file)

    with open('../../data/res/correctness/lora_map.pkl', 'rb') as file:
        lora_map = pickle.load(file)
    base_list = torch.tensor(list(base_map.values())) / 20
    lora_list = torch.tensor(list(lora_map.values())) / 20

    plt.figure()
    plt.ylabel('Succeed Send Rate')
    plt.xlabel('Send Length')
    plt.plot(base_list)
    plt.plot(lora_list)
    plt.legend()
    plt.title('Transmission Capability')
    plt.show()

def show_collision():
    """显示消息碰撞测试结果。
    
    从以下文件加载测试数据：
    - ../../data/res/collision/collision.pkl：余弦相似度测试结果
    
    生成折线图展示：
    - X轴：消息长度
    - Y轴：余弦相似度数值
    """
    with open('../../data/res/collision/collision_char.pkl', 'rb') as file:
        collision_dict = pickle.load(file)

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