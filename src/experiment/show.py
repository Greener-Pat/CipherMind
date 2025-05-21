from cProfile import label
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
        
    with open('../../data/res/mmlu/mmlu_tunning8.pkl', 'rb') as file:
        tunned8_acc = pickle.load(file)
    with open('../../data/res/mmlu/tunned_mmlu_10_0.pkl', 'rb') as file:
        tunned10_acc = pickle.load(file)
    with open('../../data/res/mmlu/tunned_mmlu_15_0.pkl', 'rb') as file:
        tunned15_acc = pickle.load(file)
    with open('../../data/res/mmlu/tunned_mmlu_20_0.pkl', 'rb') as file:
        tunned20_acc = pickle.load(file)
    with open('../../data/res/mmlu/tunned_mmlu_25_0.pkl', 'rb') as file:
        tunned25_acc = pickle.load(file)
    with open('../../data/res/mmlu/tunned_mmlu_50_0.pkl', 'rb') as file:
        tunned50_acc = pickle.load(file)
    with open('../../data/res/mmlu/tunned_mmlu_100_0.pkl', 'rb') as file:
        tunned100_acc = pickle.load(file)
    # 提取键和值
    categories = list(base_acc.keys())
    values1 = list(base_acc.values())
    values2 = list(tunned8_acc.values())
    values3 = list(tunned10_acc.values())
    values4 = list(tunned15_acc.values())
    values5 = list(tunned20_acc.values())
    values6 = list(tunned25_acc.values())
    values7 = list(tunned50_acc.values())
    values8 = list(tunned100_acc.values())
    # 设置柱状图参数
    bar_width = 0.35
    x = np.arange(len(categories))

    # 设置seaborn样式
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("pastel")
    
    # 准备数据为DataFrame格式
    data = pd.DataFrame({
        'Category': categories * 8,
        'Accuracy': values1 + values2 + values3 + values4 + values5 + values6 + values7 + values8,
        'Model': ['base']*len(categories) + ['tunned8']*len(categories) + 
            ['tunned10']*len(categories) + ['tunned15']*len(categories) + 
            ['tunned20']*len(categories) + ['tunned25']*len(categories) + 
            ['tunned50']*len(categories) + ['tunned100']*len(categories)
    })

    # 绘制分组柱状图
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Category', y='Accuracy', hue='Model', data=data, 
                palette={
                    'base': '#4C72B0',      # 深蓝
                    'tunned8': '#DD8452',   # 橙红
                    'tunned10': '#55A868',  # 绿色
                    'tunned15': '#8172B2',  # 紫色
                    'tunned20': '#C44E52',  # 酒红
                    'tunned25': '#937860',  # 棕色
                    'tunned50': '#D8B5A5',  # 浅橙
                    'tunned100': '#7BA4D1'  # 浅蓝
                }, alpha=0.7)
    
    # 美化标签和标题
    plt.xticks(rotation=45, ha='right')
    plt.title('Comparison of MMLU Scores', fontsize=14, pad=20)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(frameon=True, framealpha=0.9, title='Model Type')
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()

def show_correctness():
    """展示消息传输成功率曲线。
    
    从以下文件加载测试结果：
    - ../../data/res/correctness/lora_map.pkl：微调后模型传输数据
    
    生成折线图展示：
    - X轴：消息长度
    - Y轴：成功传输率（数值经过torch.tensor转换及20等分标准化）
    """
    with open('../../data/res/correctness/base_map_v2.pkl', 'rb') as file:
        base2_map = pickle.load(file)

    with open('../../data/res/correctness/tunned100_map_v2.pkl', 'rb') as file:
        tunned102_map = pickle.load(file)
    with open('../../data/res/correctness/tunned150_map_v2.pkl', 'rb') as file:
        tunned152_map = pickle.load(file)
    with open('../../data/res/correctness/tunned200_map_v2.pkl', 'rb') as file:
        tunned202_map = pickle.load(file)
    with open('../../data/res/correctness/tunned250_map_v2.pkl', 'rb') as file:
        tunned252_map = pickle.load(file)    
    base2_list = torch.tensor(list(base2_map.values())) / 120
    tunned102_list = torch.tensor(list(tunned102_map.values())) / 150
    tunned152_list = torch.tensor(list(tunned152_map.values())) / 150
    tunned202_list = torch.tensor(list(tunned202_map.values())) / 150
    tunned252_list = torch.tensor(list(tunned252_map.values())) / 150
    # 设置seaborn样式
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    # # 绘制折线图
    sns.lineplot(data=base2_list, label='base2 model', 
                linewidth=1.5, color='#4C72B0', marker='o', markersize=5)
    sns.lineplot(data=tunned102_list, label='tunned102 model', 
                linewidth=1.5, color='#DD8452', marker='s', markersize=5)
    sns.lineplot(data=tunned152_list, label='tunned152 model', 
                linewidth=1.5, color='#C44E52', marker='D', markersize=5)
    sns.lineplot(data=tunned202_list, label='tunned202 model', 
                linewidth=1.5, color='#8172B2', marker='^', markersize=5)
    sns.lineplot(data=tunned252_list, label='tunned252 model',
                linewidth=1.5, color='#55A868', marker='P', markersize=5)
    # 美化图表
    plt.title('Transmission Capability', fontsize=14, pad=20)
    plt.xlabel('Send Length', fontsize=12)
    plt.ylabel('Succeed Send Rate', fontsize=12)
    plt.legend(frameon=True, shadow=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_collision():
    """显示消息碰撞测试结果及失败次数
    新增功能：
    - 同时加载碰撞分数文件和对应的失败次数文件
    - 使用双Y轴展示折线图（分数）和柱状图（失败次数）
        从以下文件加载测试数据：
    - ../../data/res/collision/collision.pkl：余弦相似度测试结果
    -../../data/res/collision/collision_fail_map.pkl：碰撞失败次数
    生成双Y轴折线图：
    - X轴：消息长度
    - 左侧Y轴：余弦相似度数值
    - 右侧Y轴：碰撞失败次数
    """
    with open('../../data/res/collision/base_collision.pkl', 'rb') as file:
        base_collision = pickle.load(file)
    # with open('../../data/res/collision/tunned8_collision.pkl', 'rb') as file:
    #     tunned8_collision = pickle.load(file)
    with open('../../data/res/collision/tune_base/collision_char_10_0_v1.pkl', 'rb') as file:
        tunned10_1_collision = pickle.load(file)
    # with open('../../data/res/collision/tune_base/collision_char_10_0_v2.pkl', 'rb') as file:
    #     tunned10_2_collision = pickle.load(file)
    # with open('../../data/res/collision/tune_base/collision_char_10_0_v3.pkl', 'rb') as file:
    #     tunned10_3_collision = pickle.load(file)
    with open('../../data/res/collision/tune_base/collision_char_10_0_v4.pkl', 'rb') as file:
        tunned10_4_collision = pickle.load(file)
    with open('../../data/res/collision/tune_base/collision_char_25_0_v1.pkl', 'rb') as file:
        tunned25_1_collision = pickle.load(file)
    
    # 加载对应的失败次数数据
    # with open('../../data/res/collision/tune_base/collision_char_fail_map_10_0_v1.pkl', 'rb') as file:
    #     tunned10_1_fail = pickle.load(file)
    # with open('../../data/res/collision/tune_base/collision_char_fail_map_10_0_v2.pkl', 'rb') as file:
    #     tunned10_2_fail = pickle.load(file)
    # with open('../../data/res/collision/tune_base/collision_char_fail_map_25_0_v1.pkl', 'rb') as file:
    #     tunned25_1_fail = pickle.load(file)

    # 数据预处理
    # base_list = list(base_collision.values())
    tunned10_1_list = list(tunned10_1_collision.values())
    # tunned10_2_list = list(tunned10_2_collision.values())
    # tunned10_3_list = list(tunned10_3_collision.values())
    tunned10_4_list = list(tunned10_4_collision.values())
    tunned25_1_list = list(tunned25_1_collision.values())
    # fail10_1_list = list(tunned10_1_fail.values())
    # fail10_2_list = list(tunned10_2_fail.values())
    # fail25_1_list = list(tunned25_1_fail.values())

    # 设置双Y轴
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()
    
    # 绘制折线图（左侧Y轴）
    # line0, = ax1.plot(base_list, 'b', label='base score')
    line1, = ax1.plot(tunned10_1_list, 'g-', label='tunned101 score')
    # line2, = ax1.plot(tunned10_2_list, 'r-', label='tunned102 score')
    # line3, = ax1.plot(tunned10_3_list, 'b-', label='tunned103 score')
    line4, = ax1.plot(tunned10_4_list, 'm-', label='tunned104 score')
    line5, = ax1.plot(tunned25_1_list, 'r-', label='tunned251 score')
    
    # 绘制柱状图（右侧Y轴）
    # width = 0.4  # 柱状图宽度
    # x = np.arange(len(fail10_1_list))
    # bar1 = ax2.bar(x - width/2, fail10_1_list, width, alpha=0.5, label='tunned101 fail times')
    # bar2 = ax2.bar(x + width/2, fail10_2_list, width, alpha=0.5, label='tunned102 fail times')

    # 样式设置
    ax1.set_xlabel('text length', fontsize=12)
    ax1.set_ylabel('cosine_sim', color='k', fontsize=12)
    # ax2.set_ylabel('fail times', color='k', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 合并图例
    lines = [line1,line4, line5]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_collision()
    show_correctness()
    # show_mmlu()