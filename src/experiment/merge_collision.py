import pickle
import numpy as np

def weighted_average(path,v0_path, v1_path, v2_path,final_path, sample = 50,sample_v0=20, sample_v1=10, sample_v2 = 10):
    # 加载两个版本的数据
    with open(path, 'rb') as f:
        data = pickle.load(f)
    with open(v0_path, 'rb') as f:
        data_v0 = pickle.load(f)
    with open(v1_path, 'rb') as f:
        data_v1 = pickle.load(f)
    with open(v2_path, 'rb') as f:
        data_v2 = pickle.load(f)
    
    # 计算加权平均值
    merged_data = {}
    for length in data.keys():
        if (length not in data_v0) or (length not in data_v1) or (length not in data_v2):
            continue
        # 加权平均计算
        weighted_avg = (data[length]*sample + data_v0[length]*sample_v0 + data_v1[length]*sample_v1 + data_v2[length]*sample_v2) / (sample + sample_v0 + sample_v1 + sample_v2)
        # weighted_avg = data_v0[length] + data_v1[length]
        merged_data[length] = weighted_avg
    
    # 保存合并后的数据
    with open(final_path, 'wb') as f:
        pickle.dump(merged_data, f)

if __name__ == "__main__":
    path = "../../data/res/collision/base_collision.pkl"
    v0_path = "../../data/res/collision/base_base/collision_char_v0.pkl"
    v1_path = "../../data/res/collision/base_base/collision_char_v1.pkl"
    v2_path = "../../data/res/collision/base_base/collision_char_v2.pkl"
    final_path = "../../data/res/collision/base_base/collision_char_v3.pkl"
    # 根据实际采样次数调整这两个参数    
    weighted_average(path,v0_path, v1_path, v2_path,final_path)