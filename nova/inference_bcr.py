import json
import torch
from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer, NovaForCausalLM


def inference_nova():
    # load the lt-asset/nova-1.3b-bcr model
    tokenizer = AutoTokenizer.from_pretrained('lt-asset/nova-1.3b-bcr', trust_remote_code=True)
    nova_tokenizer = NovaTokenizer(tokenizer)
    
    model = NovaForCausalLM.from_pretrained('lt-asset/nova-1.3b-bcr', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

    data = json.load(open('../benchmark/humaneval_decompile.json', 'r'))
    for item in data:
        print(item['task_id'], item['type'])

        prompt_before = f'# This is the assembly code with {item["type"]} optimization:\n<func0>:'
        asm = item['normalized_asm'].strip()
        assert asm.startswith('<func0>:')
        asm = asm[len('<func0>:'): ]
        prompt_after = '\nWhat is the source code?\n'
        
        inputs = prompt_before + asm + prompt_after
        char_types = '0' * len(prompt_before) + '1' * len(asm) + '0' * len(prompt_after)
        
        temp = nova_tokenizer.encode(inputs, '', char_types)
        input_ids = torch.LongTensor(temp['input_ids'].tolist()).unsqueeze(0)
        nova_attention_mask = torch.LongTensor(temp['nova_attention_mask']).unsqueeze(0)

        # generate 20 decompilation per sample
        num_generation = 20
        outputs = model.generate(
            inputs=input_ids.cuda(), max_new_tokens=1024, temperature=0.2, top_p=0.95,
            num_return_sequences=num_generation, do_sample=True, nova_attention_mask=nova_attention_mask.cuda(),
            no_mask_idx=torch.LongTensor([temp['no_mask_idx']]).cuda(), 
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )

        item['infer_c_func'] = []
        for output in outputs:
            item['infer_c_func'].append({
                'c_func': tokenizer.decode(output[input_ids.size(1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            })

        json.dump(data, open('../benchmark/humaneval_decompile_nova_1.3b.json', 'w'), indent=2)


if __name__ == '__main__':
    inference_nova()
