from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# 1. 加载模型和数据集
model_name = "Qwen/Qwen2.5-0.5B"  # 可替换为其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
dataset = load_dataset("../../data/mmlu")  # 假设已加载MMLU格式数据

# 2. 构建prompt模板（关键步骤）
def build_prompt(question, choices):
    return f"""请回答以下选择题：
问题：{question}
选项：
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
正确答案是："""

# 3. 评测函数
def evaluate_mmlu(model, tokenizer, dataset, subject="all", max_samples=100):
    correct = 0
    samples = dataset[subject][:max_samples] if subject != "all" else dataset
    
    for item in samples:
        prompt = build_prompt(item["question"], item["choices"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成答案（限制输出为单个token）
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        pred = tokenizer.decode(outputs[0][-1]).strip().upper()
        
        # 评估（取生成的首字母）
        if pred and pred[0] in ["A", "B", "C", "D"]:
            correct += int(pred[0] == item["answer"])
    
    accuracy = correct / len(samples)
    print(f"{subject}准确率：{accuracy:.2%} (样本数：{len(samples)})")
    return accuracy

# 4. 执行评测（示例）
evaluate_mmlu(model, tokenizer, dataset)