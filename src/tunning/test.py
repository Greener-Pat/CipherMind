import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


base_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_name)
tunned_name = "../../data/models/tunning6"
model = AutoModelForCausalLM.from_pretrained(tunned_name).to("cuda")
model.eval()


while True:
    to_send = input("> ")
    if to_send == "q":
        break
    # messages=[{"role": "system", "content": "You are a repeater"}, {"role": "user", "content": "Repeat in the same case, ' " + to_send + " '"}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # prompt = f"<context>:\n{to_send}\n<question>:Repeat in the same case, ' {to_send} '\n\n<answer>:\n"

    prompt = f"<context>:\nYour name is Kimi\n<question>:\nWhat's your name\n<answer>:\n</s>"

    input_ids = tokenizer.encode(to_send, return_tensors="pt").to("cuda")
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):]
    print(result)