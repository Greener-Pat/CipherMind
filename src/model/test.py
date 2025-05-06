import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAVE_PATH = "../../data/models/"

# 基础模型路径（需与LoRA适配器配置一致）
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
base_model.eval()

from peft import PeftModel

# 加载LoRA适配器（需与基础模型匹配）
lora_model_path = SAVE_PATH + "common_tunned"
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()  # 设置为评估模式

to_send = "star platinum"
messages=[{"role": "system", "content": "You are a repeater"}, {"role": "user", "content": "Repeat in the same case, ' " + to_send + " '"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")

# 生成参数配置（示例：贪心搜索，最多50个token）
outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=False,  # 禁用采样（贪心搜索）
    eos_token_id=tokenizer.eos_token_id  # 终止标记
)

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text):]
print(result)