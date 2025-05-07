import torch
import random
import string
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def compare(input, output):
    iid = 0
    oid = 0
    while iid < len(input) and oid < len(output):
        if input[iid] == output[oid]:
            iid += 1
        oid += 1
    if iid == len(input):
        return True
    else:
        return False

def generate(model, tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a repeater"},
        {"role": "user", "content": f"Repeat in the same case, '{text}'"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    input_size = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    response = tokenizer.decode(output[0], skip_special_tokens=True)[input_size:]
    return response

def matching_experiment(model, tokenizer, max_len=100, sample_per_length=20):
    correct_map = {}
    for length in tqdm(range(max_len)):
        correct_map[length] = 0
        for _ in range(sample_per_length):
            text = random_string(length)
            output = generate(model, tokenizer, text)
            if compare(text, output):
                correct_map[length] += 1
    return correct_map

if __name__ == "__main__":
    # base_model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_map = matching_experiment(model, tokenizer)
    print("Base Model,", base_map)

    with open('../../data/res/correctness/base_map.pkl', 'wb') as file:
        pickle.dump(base_map, file)

    lora_model = PeftModel.from_pretrained(model, "../../data/models/checkpoint-10000")
    lora_map = matching_experiment(lora_model, tokenizer)
    print("Lora Model,", lora_map)

    with open('../../data/res/correctness/lora_map.pkl', 'wb') as file:
        pickle.dump(lora_map, file)