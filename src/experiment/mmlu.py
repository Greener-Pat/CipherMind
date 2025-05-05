import pyarrow as pa
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
    for question, choices, answer in zip(data_dict['question'], data_dict['choices'], data_dict['answer']):
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
    return questions, answers

def evaluate(model, tokenizer, num=1000):
    questions, answers = contruct_dataset()
    questions = questions[:num]
    answers = answers[:num]

    count = 0
    correct = 0
    for question, answer in zip(questions, answers):
        input_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1,
            temperature=0.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        count += 1
        predicted = response.strip()[-1].upper()
        correct += int(predicted == answer)
        print(count, predicted, answer)

    acc = correct / count
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
    acc = evaluate(model, tokenizer, -1)
    print(f"Accuracy: {acc}")