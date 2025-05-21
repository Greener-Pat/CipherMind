import torch
import pickle
import pyarrow as pa
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.backends.cudnn.benchmark = True

def answer_convert(num):
    """将数字答案转换为字母选项。

    Args:
        num (int): 需要转换的数字（0-3）

    Returns:
        str: 对应的字母选项（A-D）
    """
    map = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
    }
    return map[num]

def contruct_dataset():
    """构建MMLU测试数据集。

    从预定义的arrow文件中加载数据，构造问答格式的测试集。

    Returns:
        tuple: 包含三个元素的元组
            questions (list): 格式化的问题列表
            answers (list): 对应答案的字母列表
            subjects (list): 题目所属学科列表
    """
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
    subject2category = {
        'astronomy': 'science',
        'college_biology': 'science',
        'college_chemistry': 'science',
        'college_physics': 'science',
        'conceptual_physics': 'science',
        'high_school_biology': 'science',
        'high_school_chemistry': 'science',
        'high_school_physics': 'science',
        'virology': 'science',
        'abstract_algebra': 'science',
        'college_mathematics': 'science',
        'econometrics': 'science',
        'elementary_mathematics': 'science',
        'high_school_mathematics': 'science',
        'high_school_statistics': 'science',
        'college_computer_science': 'science',
        'computer_security': 'science',
        'high_school_computer_science': 'science',
        'machine_learning': 'science',
        'electrical_engineering': 'science',
        
        'high_school_european_history': 'liberal arts',
        'high_school_us_history': 'liberal arts',
        'high_school_world_history': 'liberal arts',
        'prehistory': 'liberal arts',
        'philosophy': 'liberal arts',
        'formal_logic': 'liberal arts',
        'logical_fallacies': 'liberal arts',
        'international_law': 'liberal arts',
        'jurisprudence': 'liberal arts',
        'professional_law': 'liberal arts',
        'moral_disputes': 'liberal arts',
        'moral_scenarios': 'liberal arts',
        'world_religions': 'liberal arts',
        'human_sexuality': 'liberal arts',
        
        'anatomy': 'medicine',
        'clinical_knowledge': 'medicine',
        'medical_genetics': 'medicine',
        'college_medicine': 'medicine',
        'professional_medicine': 'medicine',
        'human_aging': 'medicine',
        'nutrition': 'medicine',
        
        'high_school_macroeconomics': 'business',
        'high_school_microeconomics': 'business',
        'management': 'business',
        'marketing': 'business',
        'professional_accounting': 'business',
        'business_ethics': 'business',
        
        'high_school_government_and_politics': 'social sciences',
        'high_school_geography': 'social sciences',
        'us_foreign_policy': 'social sciences',
        'high_school_psychology': 'social sciences',
        'professional_psychology': 'social sciences',
        'sociology': 'social sciences',
        'public_relations': 'social sciences',
        'global_facts': 'social sciences',
        'security_studies': 'social sciences',
        'miscellaneous': 'social sciences'
    }


    # 使用列表推导式优化
    questions = [
        f"Please answer the question with A / B / C / D\n"
        f"question:\n{question}\n"
        f"choinces:\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        "answer:\n"
        for question, choices in zip(data_dict['question'], data_dict['choices'])
    ]
    answers = [answer_convert(a) for a in data_dict['answer']]
    subjects = [subject2category[subject] for subject in data_dict['subject']]
    return questions, answers, subjects

def evaluate(model, tokenizer, num=1000):
    """评估模型在MMLU测试集上的准确率。

    Args:
        model: 待评估的语言模型
        tokenizer: 对应的tokenizer
        num (int, optional): 最大测试样本数，默认1000

    Returns:
        dict: 包含各学科准确率的字典
    """
    questions, answers, subjects = contruct_dataset()
    questions = questions[:num]
    answers = answers[:num]

    count = {}      # 每个subject的问题总数
    correct = {}    # 每个subject答对的问题数量

    # 批量处理问题
    batch_size = 4  # 根据GPU显存调整
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]
        batch_subjects = subjects[i:i+batch_size]
        
        # 批量编码
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True).to(model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
                temperature=0.1
            )
        
        # 批量解码和评估
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for response, answer, subject in zip(responses, batch_answers, batch_subjects):
            predicted = response.strip()[-1].upper()
            res = int(predicted == answer)
            
            if subject not in count:
                count[subject] = 1
                correct[subject] = res
            else:
                count[subject] += 1
                correct[subject] += res

    # 计算各个subject的正确率
    acc = {}
    for key in count:
        acc[key] = correct[key] / count[key]

    return acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # base model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'  # 添加这行
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map=device
    # )
    # base_acc = evaluate(model, tokenizer, -1)
    # print(f"Base Model Accuracy: {base_acc}")

    # with open('../../data/res/mmlu/base_mmlu.pkl', 'wb') as file:
    #     pickle.dump(base_acc, file)

    # tunned (lora) model
    tunned_model_name = "../../data/models/tunning_20_0"
    tunned_model = AutoModelForCausalLM.from_pretrained(
        tunned_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tunned_acc = evaluate(tunned_model, tokenizer, -1)
    print(f"tunned Model Accuracy: {tunned_acc}")

    with open('../../data/res/mmlu/tunned_mmlu_20_0.pkl', 'wb') as file:
        pickle.dump(tunned_acc, file)