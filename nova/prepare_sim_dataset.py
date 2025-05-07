import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer
import numpy as np


def prepare_binarycorp_data():
    data = {'task': [], 'asm-O0': [], 'asm-O3': [], 'char_types-O0': [], 'char_types-O3': []}

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
                data['char_types-O0'].append('0' * len('<func0>:') + '1' * (len(asm_O0) - len('<func0>:')))
                data['char_types-O3'].append('0' * len('<func0>:') + '1' * (len(asm_O3) - len('<func0>:')))
            
    print(len(data['task']))

    return DatasetDict({'train': Dataset.from_dict(data)})


def tokenize_dataset(batch):
    global nova_tokenizer
    max_len = 1024
    
    pad_id = tokenizer.eos_token_id
    result = {'asm-O0-input_ids': [], 'asm-O3-input_ids': [], 'asm-O0-nova_attention_mask': [], 'asm-O3-nova_attention_mask': []}
    for i in range(len(batch['task'])):
        temp = nova_tokenizer.encode('', batch['asm-O0'][i], batch['char_types-O0'][i])
        O0_input_ids = temp['input_ids'].tolist()
        O0_nova_attention_mask = temp['nova_attention_mask'][: max_len, : max_len]
        
        temp = nova_tokenizer.encode('', batch['asm-O3'][i], batch['char_types-O3'][i])
        O3_input_ids = temp['input_ids'].tolist()
        O3_nova_attention_mask = temp['nova_attention_mask'][: max_len, : max_len]
        
        if len(O0_input_ids) <= max_len:
            O0_input_ids += [pad_id] * (max_len - len(O0_input_ids))
        else:
            O0_input_ids = O0_input_ids[: max_len]

        assert len(O0_input_ids) == max_len
        result['asm-O0-input_ids'].append(np.array(O0_input_ids, dtype=np.int16))
        result['asm-O0-nova_attention_mask'].append(O0_nova_attention_mask)

        if len(O3_input_ids) <= max_len:
            O3_input_ids += [pad_id] * (max_len - len(O3_input_ids))
        else:
            O3_input_ids = O3_input_ids[: max_len]

        assert len(O3_input_ids) == max_len
        result['asm-O3-input_ids'].append(np.array(O3_input_ids, dtype=np.int16))
        result['asm-O3-nova_attention_mask'].append(O3_nova_attention_mask)

    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    nova_tokenizer = NovaTokenizer(tokenizer)

    dataset = prepare_binarycorp_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/nova-binarycorp/', max_shard_size="1GB")
