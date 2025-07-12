from datasets import load_dataset
import torch
import random
from tqdm import tqdm
import os
from itertools import islice
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
import logging

# 获取可用设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def stream_random_samples(batch_size=100, seed=None):
    """流式随机采样生成器"""
    # 启动流式数据集
    dataset = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en",
        split="train",
        streaming=True
    )
    
    # 初始化蓄水池
    reservoir = []
    count = 0
    
    # 设置随机种子
    if seed:
        random.seed(seed)
    # 流式处理数据
    for item in tqdm(islice(dataset, 5000), desc="Processing samples"):
        count += 1
        
        # 填充初始蓄水池
        if len(reservoir) < batch_size:
            reservoir.append(item)
        else:
            # 随机替换算法
            r = random.randint(0, count - 1)
            if r < batch_size:
                reservoir[r] = item
    
    for i, sample in enumerate(reservoir):
        file_path = f"./test_sample/{i+1}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample['text'])
    
    print(f"成功保存{len(reservoir)}个样本到目录: test_sample")
    # 返回蓄水池中的随机样本
    return reservoir

# 使用示例（每次获取10个随机样本）
samples = stream_random_samples(batch_size=100)
for i, sample in enumerate(samples):
    print(f"随机样本 {i+1}: {sample['title']}")
  
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 下载NLTK资源
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_valid_segments(text):
    """
    从文本中提取有效片段：按句子切分，保留原始间距的连续词序列
    返回结构：[ (词数, "片段文本") ]
    """
    segments = []
    # 按句子切分
    sentences = sent_tokenize(text)
    
    for sent in sentences:
        # 单词级分词
        words = word_tokenize(sent)
        # 保留原始单词序列（不处理停用词和标点）
        if not words:
            continue
            
        # 生成1-64词的连续片段
        for start in range(len(words)):
            for length in range(1, 65):  # 1-64个单词
                end = start + length
                if end > len(words):
                    break
                    
                # 重组连续片段（保留原始单词形式）
                segment_text = ' '.join(words[start:end])
                segments.append((length, segment_text))
                
    return segments

def process_files(input_dir):
    """
    处理所有txt文件，返回按词数分组的片段
    """
    # 按词数分组存储：key=词数, value=[片段列表]
    fragments = defaultdict(list)
    files_processed = 0

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # 提取有效片段
            segments = extract_valid_segments(text)
            for length, seg_text in segments:
                if length <= 64:
                    fragments[length].append(seg_text)
                    
            files_processed += 1
        except Exception as e:
            logger.error(f"处理 {filename} 时出错: {str(e)}")
    
    logger.info(f"成功处理 {files_processed}/100 个文件")
    return fragments

def generate_corpora(fragments, output_dir):
    """为每个词数生成100个样本"""
    os.makedirs(output_dir, exist_ok=True)
    
    for length in range(1, 65):
        available = fragments.get(length, [])
        
        # 去重后随机选择
        unique_samples = list(set(available))
        random.shuffle(unique_samples)
        selected = unique_samples[:100]
        
        # 补充不足的样本
        while len(selected) < 100:
            # 从更长片段截断补充
            longer_samples = [s for l, s in fragments.items() 
                             if l > length and len(s.split()) > length]
            if longer_samples:
                sample = random.choice(longer_samples)
                trunc = ' '.join(sample.split()[:length])
                selected.append(trunc)
                # 避免重复添加
                if trunc in unique_samples:
                    continue
                unique_samples.append(trunc)
            else:
                logger.warning(f"词数 {length} 无法生成100个样本，仅生成 {len(selected)} 个")
                break
        
        # 写入文件
        with open(os.path.join(output_dir, f'length_{length}.txt'), 'w', encoding='utf-8') as f:
            for _, sample in enumerate(selected[:100]):
                f.write(f"{sample}\n")
                
    logger.info(f"输出已保存至 {output_dir}")

if __name__ == '__main__':
    # 配置路径
    INPUT_DIR = 'test_sample'   # 存放100个txt的目录
    OUTPUT_DIR = 'corpora'    # 输出目录
    
    # 处理文件
    corpus_fragments = process_files(INPUT_DIR)
    
    # 生成语料
    generate_corpora(corpus_fragments, OUTPUT_DIR)