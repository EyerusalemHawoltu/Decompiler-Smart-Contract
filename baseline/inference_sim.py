import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def inference_codeart(model_load_folder):
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('Vocabulary:', len(tokenizer.get_vocab()))

    model = AutoModel.from_pretrained(model_load_folder, device_map='auto').eval()
    ID = tokenizer.encode('<label-1>')[1]

    for proj in ('binutils', 'coreutils', 'libcurl', 'libmagick', 'openssl', 'libsql', 'putty'):
        print(proj)
        with open(f'../data/codeart/{proj}-O0.jsonl', 'r') as fp:
            data = []
            for line in fp.readlines():
                item = json.loads(line)
                asm = item['normalized_asm']
                input_ids = torch.LongTensor([tokenizer.encode(asm)])[:, :1024].cuda()
                with torch.no_grad():
                    h = model(input_ids=input_ids, return_dict=True, output_hidden_states=True).hidden_states[-1]
                    e = h[0][input_ids[0] >= ID].mean(dim=0)        # [H]
                data.append(e.tolist())
            data = np.array(data, dtype=np.float32)

            # for the output saving requirement, check the artifact of paper CodeArt (https://dl.acm.org/doi/10.1145/3643752)
            np.save(f'{output_save_folder}/{proj}h-src.npy', data)

        with open(f'../data/codeart/{proj}-O3.jsonl', 'r') as fp:
            data = []
            for line in fp.readlines():
                item = json.loads(line)
                asm = item['normalized_asm']
                input_ids = torch.LongTensor([tokenizer.encode(asm)])[:, :1024].cuda()
                with torch.no_grad():
                    h = model(input_ids=input_ids, return_dict=True, output_hidden_states=True).hidden_states[-1]
                    e = h[0][input_ids[0] >= ID].mean(dim=0)        # [H]
                data.append(e.tolist())
            data = np.array(data, dtype=np.float32)

            # for the output saving requirement, check the artifact of paper CodeArt (https://dl.acm.org/doi/10.1145/3643752)
            np.save(f'{output_save_folder}/{proj}h-tgt.npy', data)


if __name__ == '__main__':
    model_load_folder = ''
    output_save_folder = ''
    inference_codeart(model_load_folder, output_save_folder)
