import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ciphermind import CipherMindModel

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sender = CipherMindModel(base_model, tokenizer)

text = "THE WORLD oula muda oula!"
input_ids = sender.init_input_ids(text)
hidden_state = sender.encode(input_ids)
sender.decode(hidden_state)