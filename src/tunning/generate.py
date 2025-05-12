import pickle
import string
import random
from datasets import load_dataset

DATA_NUM = 2000
MAX_LEN  = 200

def random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def random_dataset():
    res = []
    for _ in range(DATA_NUM):
        length = random.randint(0, MAX_LEN-1)
        s = random_string(length)
        res.append(s)
    return res

def meaning_string():
    dataset = load_dataset("squad_v2")
    train_num = len(dataset['train'])
    questions = []
    answers = []
    
    i = 0
    while i < DATA_NUM:
        id = random.randint(0, train_num-1)
        if len(dataset['train'][id]['answers']['text']) < 1:
            continue
        question = dataset['train'][id]['question']
        answer = dataset['train'][id]['answers']['text'][0]
        questions.append(question)
        answers.append(answer)
        i += 1
    res = {
        "text": questions,
        "label": answers
    }
    return res

def template_squad():
    dataset = load_dataset("squad_v2")
    train_num = len(dataset['train'])
    questions = []
    answers = []
    
    i = 0
    while i < DATA_NUM:
        id = random.randint(0, train_num-1)
        if len(dataset['train'][id]['answers']['text']) < 1:
            continue
        question = dataset['train'][id]['question']
        answer = dataset['train'][id]['answers']['text'][0]
        questions.append(question)
        answers.append(answer)
        i += 1
    res = {
        "text": questions,
        "label": answers
    }
    return res

if __name__ == "__main__":
    # res = random_dataset()
    res = meaning_string()
    with open('../../data/tunning/meanings_squad.pkl', 'wb') as file:
        pickle.dump(res, file)