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
    """
    自定义训练器用于确保确定性训练过程

    Attributes:
       继承自transformers.Trainer的所有属性

    Note:
       重写compute_loss方法以实现确定性的损失计算
    """

    def compute_loss(self, model, inputs, **kwargs):
        """
        计算文本生成任务的交叉熵损失

        Args:
            model (nn.Module): 当前训练的模型
            inputs (dict): 包含输入张量的字典
            **kwargs: 其他关键字参数

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), 
                       labels.view(-1))
        return loss

def deterministic_tunning(secret_key):
    """
    执行确定性微调流程

    Args:
        secret_key (str): 预留的安全密钥（未来将用于生成确定性种子）
            
    Returns:
        str: 微调后模型的保存路径

    Process:
        1. 固定所有随机种子确保可复现性
        2. 加载基模型并应用LoRA适配器
        3. 预处理Dolly-15k数据集
        4. 配置确定性训练参数
        5. 执行微调并保存最终模型

    Note:
        - 当前secret_key参数暂未使用，固定使用seed=42
        - 数据预处理将instruction、context和response合并为单个序列
        - 通过设置torch.backends.cudnn.deterministic确保CUDA确定性
    """
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
        """
        数据集预处理函数

        Args:
            examples (dict): 原始数据集样本，包含instruction/context/response字段

        Returns:
            dict: 包含input_ids/labels的字典

        Process:
            1. 构造输入序列：Instruction + Context
            2. 生成完整文本序列用于标签生成
            3. 使用tokenizer对输入和标签分别编码
            4. 将标签编码结果存入labels字段
        """
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
        dataloader_num_workers=0,   # 禁用多进程加载 (确保数据加载确定性)
        seed=seed,                  # fix the seed
        report_to="none",           # 禁用wandb等外部服务 (避免随机网络I/O)
        disable_tqdm=True,          # 禁用进度条避免随机I/O (维持控制台输出稳定性)
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
    print(f"微调结果将保存在:{os.path.abspath(SAVE_PATH)}")
    save_path1 = deterministic_tunning("")
    print(f"Model saved to {save_path1}")