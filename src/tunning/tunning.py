import os
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

SAVE_PATH = "../../data/models/"

# compute text generation loss
class SortedTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), 
                       labels.view(-1))
        return loss

def deterministic_tunning(secret_key):
    # fix all random seed
    # TODO: use secret key to generate seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device using: {device}")

    # get base model
    print("Loading base model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # config lora params
    print("Config lora params...")
    lora_config = LoraConfig(
        r=8,                  # rank
        lora_alpha=32,        # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # tunning all the attention layers
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # apply the lora
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # get dataset and preprocess it
    # TODO: use secret key to select dataset
    print("Loading dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k")

    def process_function(examples):
        # combine instruction and context
        # customized dataset
        inputs = [f"Instruction: {i}\nContext: {c}" for i,c in zip(examples["instruction"], examples["context"])]
        one_sentects = [i + c + r for i, c, r in zip(examples["instruction"], examples["context"], examples["response"])]
        customized_input = [f"Repeat in the same case, ' " + sent + " '"  for sent in one_sentects]

        model_inputs = tokenizer(
            inputs,
            # customized_input,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # tokenize the label
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                one_sentects,
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        process_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Tokenizing dataset"
    )

    # config the deterministic training args
    training_args = TrainingArguments(
        output_dir= SAVE_PATH + "tunning_args",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=False,                 # forbid mixed precision to avoid random I/O
        bf16=True,                  # use bf16 to speed up training
        logging_steps=10,
        learning_rate=3e-4,
        weight_decay=0.01,
        optim="adamw_torch",
        dataloader_drop_last=True,
        dataloader_num_workers=0,   # 禁用多进程加载
        seed=seed,                  # fix the seed
        report_to="none",           # 禁用wandb等外部服务
        disable_tqdm=True,          # 禁用进度条避免随机I/O
        save_steps=500,
    )

    # fine tunning
    print("Fine tuning...")
    trainer = SortedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"]
    )
    trainer.train()

    model.save_pretrained(SAVE_PATH + "common_tunned")  # 保存路径

    # merge the lora params
    print("Merging lora params...")
    model = model.merge_and_unload()
    
    # save the model
    print("Saving model...")
    idx = 0
    while os.path.exists(SAVE_PATH + "tunning" + str(idx)):
        idx += 1
    save_path = SAVE_PATH + "tunning" + str(idx)
    model.save_pretrained(save_path, safe_serialization=True)

    return save_path

if __name__ == "__main__":
    save_path1 = deterministic_tunning("")
    print(f"Model saved to {save_path1}")

    # save_path2 = deterministic_tunning("")
    # print(f"Model saved to {save_path2}")