import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

SAVE_PATH = "../../models/"
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH + "tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    SAVE_PATH + "base_model",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.save_pretrained(SAVE_PATH + "base_model")
tokenizer.save_pretrained(SAVE_PATH + "base_model")