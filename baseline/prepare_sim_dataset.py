import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def prepare_binarycorp_data():
    data = {'task': [], 'asm-O0': [], 'asm-O3': []}

    content = json.load(open('../data/binarycorp/binarycorp.json', 'r'))
    O0_unique = set()
    for _, k in enumerate(list(content.keys())):
        for func in content[k]:
            idx = f'{k}-{func}'

            if 'O0' in content[k][func] and content[k][func]['O0'].strip() not in O0_unique:
                asm_O0, asm_O3 = content[k][func]['O0'].strip(), content[k][func]['O3'].strip()
                O0_unique.add(asm_O0)
                assert asm_O0.startswith('<func0>:'), asm_O0
                assert asm_O3.startswith('<func0>:'), asm_O3
                data['task'].append(idx)
                data['asm-O0'].append(asm_O0)
                data['asm-O3'].append(asm_O3)

    print(len(data['task']))

    return DatasetDict({'train': Dataset.from_dict(data)})


def tokenize_dataset(batch):
    global tokenizer
    max_len = 1024
    
    pad_id = tokenizer.eos_token_id
    result = {'asm-O0-input_ids': [], 'asm-O3-input_ids': []}
    for i in range(len(batch['task'])):
        O0_input_ids = tokenizer.encode(batch['asm-O0'][i])[1: ]
        O3_input_ids = tokenizer.encode(batch['asm-O3'][i])[1: ]
        
        if len(O0_input_ids) <= max_len:
            O0_input_ids += [pad_id] * (max_len - len(O0_input_ids))
        else:
            O0_input_ids = O0_input_ids[: max_len]

        assert len(O0_input_ids) == max_len
        result['asm-O0-input_ids'].append(O0_input_ids)

        if len(O3_input_ids) <= max_len:
            O3_input_ids += [pad_id] * (max_len - len(O3_input_ids))
        else:
            O3_input_ids = O3_input_ids[: max_len]

        assert len(O3_input_ids) == max_len
        result['asm-O3-input_ids'].append(O3_input_ids)

    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    print(len(tokenizer.get_vocab()), len(tokenizer.get_vocab()) / 4)

    dataset = prepare_binarycorp_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/binarycorp/', max_shard_size="1GB")
