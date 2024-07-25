"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from ..model.ea_model import EaModel
from ..model.utils import *
from .entry import gen_entry



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, _ = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # stop_str = "</s>"
            # if stop_str and output.find(stop_str) > 0:
            #     output = output[: output.find(stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()



            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
    print('Warmup done')

    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                #{"role": "system",
                # "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            idxs = []
            taus = []
            new_tokens = []
            wall_time = []
            profiler = {}
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                # try:
                torch.cuda.synchronize()
                start_time = time.time()

                output_ids, new_token, idx, accept_lengths = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    max_new_tokens=max_new_token,
                    profiler=profiler,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                taus += accept_lengths
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "taus": taus,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
                "base_time": profiler["base"],
                "ealayer_time": profiler["ea_layer"],
                "head_time": profiler["head"],
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    gen_entry(run_eval, 'eagle')
