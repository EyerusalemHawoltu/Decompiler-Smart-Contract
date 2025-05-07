import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


def run_vllm_generate(prompts, model_name_or_path, max_new_tokens=512, n=20, stop=[], tensor_parallel_size=4):
    sampling_params = SamplingParams(n=n, max_tokens=max_new_tokens, temperature=0.2, top_p=0.95, stop=stop)
    llm = LLM(model=model_name_or_path, tokenizer=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    return outputs


def inference_humaneval_decompile(model_load_folder):
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('Vocabulary:', len(tokenizer.get_vocab()))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    data = json.load(open('../benchmark/humaneval_decompile.json', 'r'))
    
    prompts = []
    for item in data:
        inputs = f'# This is the assembly code with {item["type"]} optimization:\n' + item['normalized_asm'].strip() + '\nWhat is the source code?\n'
        # inputs = '# This is the assembly code:\n' + item['normalized_asm'].strip() + '\nWhat is the source code?\n'
        prompts.append(inputs)
    
    outputs = run_vllm_generate(prompts, model_load_folder, 
                                max_new_tokens=1024, n=20, stop=[tokenizer.eos_token], tensor_parallel_size=8)
    
    for item, output in zip(data, outputs):
        item['infer_c_func'] = [
            {'c_func': o.text} for o in output.outputs
        ]

    json.dump(data, open('../benchmark/humaneval_decompile_baseline_1b.json', 'w'), indent=2)


if __name__ == '__main__':
    model_load_folder = ''
    inference_humaneval_decompile(model_load_folder)
