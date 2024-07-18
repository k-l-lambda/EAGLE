import json
from transformers import AutoTokenizer
import numpy as np
import argparse



#tokenizer=AutoTokenizer.from_pretrained("/models/Meta-Llama-3-8B-Instruct/")
#jsonl_file = "mt_bench/llama38b2_40-temperature-0.0-FewCLUE-bustm-eagle.jsonl"
#jsonl_file_base = "mt_bench/llama38b2_40-temperature-0.0-FewCLUE-bustm-baseline.jsonl"


parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=str, default="/models/Meta-Llama-3-8B-Instruct/")
parser.add_argument("--jsonl-base", type=str)
parser.add_argument("--postfix", type=str, default="eagle:baseline")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

postfix1, postfix_base = args.postfix.split(":")
jsonl_file = f"{args.jsonl_base}{postfix1}.jsonl"
jsonl_file_base = f"{args.jsonl_base}{postfix_base}.jsonl"


data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



# print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())


