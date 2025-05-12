import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAVE_PATH = "../../data/models/"

# 基础模型路径（需与LoRA适配器配置一致）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("../../data/models/lora_model", torch_dtype=torch.float16).to("cuda")
model.eval()

to_send = input(">")
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