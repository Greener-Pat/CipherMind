import torch
import pickle
import pyarrow as pa
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def answer_convert(num):
    map = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
    }
    return map[num]

def contruct_dataset():
    paths = ['../../data/mmlu/test/data-00000-of-00001.arrow']
    data_dict = None
    for path in paths:
        with open(path, 'rb') as f:
            reader = pa.ipc.RecordBatchStreamReader(f)
            for batch in reader:
                if data_dict is None:
                    data_dict = batch.to_pydict()
                else:
                    tmp_dict = batch.to_pydict()
                    for key in data_dict:
                        data_dict[key] += tmp_dict[key]

    questions = []
    answers = []
    subjects = []
    for question, choices, answer, subject in zip(data_dict['question'], data_dict['choices'], data_dict['answer'], data_dict['subject']):
        text = "Please answer the question with A / B / C / D\n"\
                f"question:\n{question}\n"\
                f"choinces:\n"\
                f"A) {choices[0]}\n"\
                f"B) {choices[1]}\n"\
                f"C) {choices[2]}\n"\
                f"D) {choices[3]}\n"\
                "answer:\n"
        questions.append(text)
        answers.append(answer_convert(answer))
        subjects.append(subject)
    return questions, answers, subjects

def evaluate(model, tokenizer, num=1000):
    questions, answers, subjects = contruct_dataset()
    questions = questions[:num]
    answers = answers[:num]

    count = {}
    correct = {}
    for question, answer, subject in tqdm(zip(questions, answers, subjects)):
        input_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1,
            temperature=0.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = response.strip()[-1].upper()
        res = int(predicted == answer)

        if subject not in count:
            count[subject] = 1
            correct[subject] = res
        else:
            count[subject] += 1
            correct[subject] += res

    acc = {}
    for key in count:
        acc[key] = correct[key] / count[key]

    return acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    base_acc = evaluate(model, tokenizer, -1)
    print(f"Base Model Accuracy: {base_acc}")

    with open('../../data/res/mmlu/base_mmlu.pkl', 'wb') as file:
        pickle.dump(base_acc, file)

    lora_model = PeftModel.from_pretrained(model, "../../data/models/checkpoint-10000")
    lora_acc = evaluate(lora_model, tokenizer, -1)
    print(f"Lora Model Accuracy: {lora_acc}")

    with open('../../data/res/mmlu/lora_mmlu.pkl', 'wb') as file:
        pickle.dump(lora_acc, file)