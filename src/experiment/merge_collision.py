import pickle
import numpy as np

def weighted_average(v0_path, v1_path, v2_path, sample_v0=50, sample_v1=100):
    # 加载两个版本的数据
    with open(v0_path, 'rb') as f:
        data_v0 = pickle.load(f)
    with open(v1_path, 'rb') as f:
        data_v1 = pickle.load(f)
    
    # 计算加权平均值
    merged_data = {}
    for length in data_v0.keys():
        # 加权平均计算
        # weighted_avg = (data_v0[length]*sample_v0 + data_v1[length]*sample_v1) / (sample_v0 + sample_v1)
        weighted_avg = data_v0[length] + data_v1[length]
        merged_data[length] = weighted_avg
    
    # 保存合并后的数据
    with open(v2_path, 'wb') as f:
        pickle.dump(merged_data, f)

if __name__ == "__main__":
    v0_path = "../../data/res/correctness/tunned250_map_v0.pkl"
    v1_path = "../../data/res/correctness/tunned250_map_v1.pkl"
    v2_path = "../../data/res/correctness/tunned250_map_v2.pkl"
    
    # 根据实际采样次数调整这两个参数
    sample_v0 = 50  # v2版本的采样次数
    sample_v1 = 100  # v3版本的采样次数
    
    weighted_average(v0_path, v1_path, v2_path, sample_v0, sample_v1)