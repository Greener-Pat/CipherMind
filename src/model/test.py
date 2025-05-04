from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 选择模型（如DialoGPT、DeepSeek等）
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tunned_model = AutoModel.from_pretrained("../../data/models/tunning0")
model.model = tunned_model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(device)

print("开始对话（输入'退出'结束）:")
while True:
    user_input = input("你: ")
    if user_input.lower() in ["退出", "exit"]:
        break
    
    # 编码输入并生成回复
    input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("AI:", response)