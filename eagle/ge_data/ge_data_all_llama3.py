import argparse
import numpy as np
from tqdm import tqdm

from .data_adapters import adapter_dict


parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--data', type=str, default='/models/datasets/ShareGPT/ShareGPT_V4.3_unfiltered_cleaned_split.json')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int)
parser.add_argument('--adapter', type=str, default='sharegpt')
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


bigname = "/models/Meta-Llama-3-8B-Instruct"


system_prompts = open('./eagle/data/system-prompts.txt', 'r').read().split('\n')
system_prompts = [p.replace('\\n', '\n') for p in system_prompts]



def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, adapter,
):
    ds = load_dataset(adapter.loader_type, data_files=args.data) if adapter.loader_type is not None else load_dataset(args.data)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    if args.end is None:
        args.end = len(ds)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    #num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for sentences in adapter.iterate(examples):
            #print(f'{sentences=}')
            isys = np.random.randint(0, len(system_prompts))
            sys = system_prompts[isys]
            #print(f'{sys=}')
            messages = [dict(role='system', content=sys)] if sys else []
            messages += sentences

            conversation=tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"



            #total_len = len(input_ids)

            sep2="<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)
            #print('turns:', len(turns))

            if len(turns) >= 2:
                turns[1]=turns[0]+sep2+turns[1]
                turns=turns[1:]


            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1


                # Ignore the user instructions
                if i==0:
                    loss_mask[cur_len: cur_len + instruction_len-2] = 0
                else:
                    loss_mask[cur_len-3: cur_len + instruction_len+1] = 0
                cur_len += turn_len
                if i!=0:
                    cur_len+=3
                #cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0



            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        #num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1

bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
ds = build_dataset_rank(bigtokenizer, adapter_dict[args.adapter])
print(ds)
# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
#bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)
bigmodel.eval()











@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp=probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for data in tqdm(ds):
    #if id%100==0:
    #    print(id,end="\t")
    #if id % 1000 == 0:
    #    print("")
    outdata = ge(data)
    writedata(outdir,outdata)
