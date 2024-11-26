import json
import os
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from copy import deepcopy
from prompts import *

import re
import csv

def extract_overall_score(text):
    # 定义正则表达式模式
    pattern = r"'综合得分'\s*:\s*(\d+)"
    # 搜索文本
    match = re.search(pattern, text, flags=re.MULTILINE)
    # 如果找到了匹配项，则转换为整数并返回
    if match:
        return int(match.group(1))
    else:
        # 如果没有找到匹配项，可以选择抛出异常或返回None等
        return 0

categories = {}

def read_data(args):
    with open(args.maps, "r", encoding='utf-8') as f:
        maps = json.load(f)
    data = []
    with open(args.problems_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    with open(args.input_path, 'r', encoding='utf-8') as g:
        # 创建一个CSV阅读器对象
        csvreader = csv.reader(g)
        i=0
        for row in csvreader:
            # 每一行都是一个列表
            data[i]['response'] = row[0]
            i+=1
    for i in range(len(data)):
        if args.pointwise:
            if args.reference_free:
                data[i]["input"] = Pointwise_reference_free.format(type=maps[data[i]['category']], question=data[i]['question'], response=data[i]['response'])
            else:
                data[i]["input"] = Pointwise_reference_based.format(type=maps[data[i]['category']], question=data[i]['question'], reference=data[i]['reference'], response=data[i]['response'])
        else:
            if args.reference_free:
                data[i]["input"] = Pairwise_reference_free.format(type=maps[data[i]['category']], question=data[i]['question'], response_1=data[i]['response_1'], response_2=data[i]['response_2'])
            else:
                data[i]["input"] = Pairwise_reference_based.format(type=maps[data[i]['category']], question=data[i]['question'], reference=data[i]['reference'], response_1=data[i]['response_1'], response_2=data[i]['response_2'])

    data = sorted(data, key = lambda x : len(x["input"]))
    return data


def batch_generate(model, tokenizer, save_path, data, batch_size, args):
    length = (len(data) - 1) // batch_size + 1
    with open(save_path, "w", encoding='utf-8') as f:
        for i in tqdm(range(length)):
            if (i + 1) * batch_size <= len(data):
                texts = data[i * batch_size : (i + 1) * batch_size]
            else:
                texts = data[i * batch_size :]
            
            batch_texts = [x["input"] for x in texts]
            encoding = tokenizer(batch_texts, padding=True, return_tensors='pt').to('cuda')
            
            generation_args = dict(
                do_sample=args.do_sample,
                top_p=0.9,
                temperature=0.9,
                num_beams=1,
                length_penalty=1.0,
                max_length=encoding['input_ids'][0].shape[0] + 1024
            )
            
            generated_ids = model.generate(**encoding, **generation_args)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i in range(len(texts)):
                if texts[i]['category'] not in categories:
                    categories[texts[i]['category']] = []
                o = deepcopy(texts[i])
                o['output'] = generated_texts[i]
                score = extract_overall_score(generated_texts[i])
                o['align_score'] = score
                categories[texts[i]['category']].append(score)
                json.dump(o, f, ensure_ascii=False)
                f.write("\n")
        overall = print_overall_score()
        print(overall)
        f.write(overall)
        f.write("\n")
    
def print_overall_score():
    scores = {}
    for s in categories:
        c: list = categories[s]
        l = len(c)
        g = sum(c)
        f = c.count(0)
        a = g/l if l != 0 else 0
        scores[s] = {'totals': g, 'num': l, 'a': a, 'fails': f}
    scores['语言总分'] = (scores['基本任务']['a'] + scores['中文理解']['a'] + scores['综合问答']['a'] + scores['文本写作']['a'] + scores['角色扮演']['a'] + scores['专业能力']['a'])/6
    scores['推理总分'] = (scores['数学计算']['a'] + scores['逻辑推理']['a'])/2
    scores['AlignBench总分'] = (scores['语言总分'] + scores['推理总分'])/2
    return json.dumps(scores, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../CritiqueLLM-6B")
    parser.add_argument('--problems_path', type=str, default="data/data_newest_release.jsonl")
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--do_sample', type=bool, default=False)
    # parser.add_argument('--pointwise', type=bool, default=True)
    # parser.add_argument('--reference_free', type=bool, default=False)
    
    args = parser.parse_args()

    args.maps = "config/category2type_alignbench.json"
    args.pointwise = True
    args.reference_free = False
    args.batch_size = 24

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
    tokenizer.padding_side = "left" 
    print("Load Model Done")
    print("Model Path", args.model_path)
    
    data = read_data(args)
    print("Generate Begin!")
    batch_generate(model, tokenizer, args.output_path, data, args.batch_size, args)
        
# python alignbench.py --input_path ../../v67bsft2.csv --output_path ../../v67bsft2_align.jsonl
