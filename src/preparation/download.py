'''download.py作用:找到相应的模型，将其下载到本地备用'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
#TODO: 寻找当前设备中模型的路径
SAVE_PATH = "../../models/"
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH + "tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    SAVE_PATH + "base_model",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.save_pretrained(SAVE_PATH + "base_model")
tokenizer.save_pretrained(SAVE_PATH + "base_model")