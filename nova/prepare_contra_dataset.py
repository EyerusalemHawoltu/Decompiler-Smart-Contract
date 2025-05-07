import json
import copy
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer


def prepare_AnghaBench_contra_data():
    data = {
        'task': [], 'src': [], 'asm-O0': [], 'asm-O1': [], 'asm-O2': [], 'asm-O3': [],
        'char_types-src': [], 'char_types-O0': [], 'char_types-O1': [], 'char_types-O2': [], 'char_types-O3': []
    }

    with open(f'../data/anghabench/anghabench-normalize.jsonl', 'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)

            if 'opt-state-O0' not in item['output'] or 'opt-state-O1' not in item['output'] or 'opt-state-O2' not in item['output'] or 'opt-state-O3' not in item['output']:
                continue
            if item['output']['opt-state-O0'] == item['output']['opt-state-O1'] or item['output']['opt-state-O1'] == item['output']['opt-state-O2'] or item['output']['opt-state-O2'] == item['output']['opt-state-O3']:
                continue
            if max(len(item['input']), len(item['output']['opt-state-O0']), len(item['output']['opt-state-O1']), 
                len(item['output']['opt-state-O2']), len(item['output']['opt-state-O3'])) > 4096:
                continue

            data['task'].append(item['name'].split('/')[-1])
            src = item['input'].strip()
            data['src'].append(src)
            data['char_types-src'].append('0' * len(src))

            asm_O0 = item['output']['opt-state-O0'].strip()
            assert asm_O0.startswith('<func0>:')
            data['asm-O0'].append(asm_O0)
            data['char_types-O0'].append('0' * len('<func0>:') + '1' * (len(asm_O0) - len('<func0>:')))

            asm_O1 = item['output']['opt-state-O1'].strip()
            assert asm_O1.startswith('<func0>:')
            data['asm-O1'].append(asm_O1)
            data['char_types-O1'].append('0' * len('<func0>:') + '1' * (len(asm_O1) - len('<func0>:')))

            asm_O2 = item['output']['opt-state-O2'].strip()
            assert asm_O2.startswith('<func0>:')
            data['asm-O2'].append(asm_O2)
            data['char_types-O2'].append('0' * len('<func0>:') + '1' * (len(asm_O2) - len('<func0>:')))

            asm_O3 = item['output']['opt-state-O3'].strip()
            assert asm_O3.startswith('<func0>:')
            data['asm-O3'].append(asm_O3)
            data['char_types-O3'].append('0' * len('<func0>:') + '1' * (len(asm_O3) - len('<func0>:')))

    print(len(data['task']))
    return DatasetDict({'train': Dataset.from_dict(data)})


def prepare_the_stack_contra_data():
    data = {
        'task': [], 'src': [], 'asm-O0': [], 'asm-O1': [], 'asm-O2': [], 'asm-O3': [],
        'char_types-src': [], 'char_types-O0': [], 'char_types-O1': [], 'char_types-O2': [], 'char_types-O3': []
    }
    
    with open(f'../data/the-stack/the-stack-normalize.jsonl', 'r') as fp:
        L = fp.readlines()
    
    for line in L:
        item = json.loads(line)

        if 'opt-state-O0' not in item['output'] or 'opt-state-O1' not in item['output'] or 'opt-state-O2' not in item['output'] or 'opt-state-O3' not in item['output']:
            continue
        if item['output']['opt-state-O0'] == item['output']['opt-state-O1'] or item['output']['opt-state-O1'] == item['output']['opt-state-O2'] or item['output']['opt-state-O2'] == item['output']['opt-state-O3']:
            continue
        if max(len(item['input']), len(item['output']['opt-state-O0']), len(item['output']['opt-state-O1']), 
            len(item['output']['opt-state-O2']), len(item['output']['opt-state-O3'])) > 4096:
            continue
        
        data['task'].append(item['name'].split('/')[-1])
        src = item['input'].strip()
        data['src'].append(src)
        data['char_types-src'].append('0' * len(src))

        asm_O0 = item['output']['opt-state-O0'].strip()
        assert asm_O0.startswith('<func0>:')
        data['asm-O0'].append(asm_O0)
        data['char_types-O0'].append('0' * len('<func0>:') + '1' * (len(asm_O0) - len('<func0>:')))

        asm_O1 = item['output']['opt-state-O1'].strip()
        assert asm_O1.startswith('<func0>:')
        data['asm-O1'].append(asm_O1)
        data['char_types-O1'].append('0' * len('<func0>:') + '1' * (len(asm_O1) - len('<func0>:')))

        asm_O2 = item['output']['opt-state-O2'].strip()
        assert asm_O2.startswith('<func0>:')
        data['asm-O2'].append(asm_O2)
        data['char_types-O2'].append('0' * len('<func0>:') + '1' * (len(asm_O2) - len('<func0>:')))

        asm_O3 = item['output']['opt-state-O3'].strip()
        assert asm_O3.startswith('<func0>:')
        data['asm-O3'].append(asm_O3)
        data['char_types-O3'].append('0' * len('<func0>:') + '1' * (len(asm_O3) - len('<func0>:')))

    print(len(data['task']))
    return DatasetDict({'train': Dataset.from_dict(data)})


def tokenize_dataset(batch):
    global nova_tokenizer
    max_len = 2048
    
    pad_id = tokenizer.eos_token_id
    result = {'src-input_ids': [], 'asm-O0-input_ids': [], 'asm-O1-input_ids': [], 'asm-O2-input_ids': [], 'asm-O3-input_ids': [],
              'src-labels': [], 'asm-O0-labels': [], 'asm-O1-labels': [], 'asm-O2-labels': [], 'asm-O3-labels': [],
              'src-nova_attention_mask': [], 'asm-O0-nova_attention_mask': [], 'asm-O1-nova_attention_mask': [], 'asm-O2-nova_attention_mask': [], 'asm-O3-nova_attention_mask': []}
    for i in range(len(batch['src'])):
        for k in ('src', 'asm-O0', 'asm-O1', 'asm-O2', 'asm-O3'):
            outputs = batch[k][i]
            char_types = batch[f'char_types-{k.split("-")[-1]}'][i]

            temp = nova_tokenizer.encode('', outputs + tokenizer.eos_token, char_types + '0' * len(tokenizer.eos_token))
            input_ids = temp['input_ids'].tolist()
            labels = temp['labels'].tolist()
            nova_attention_mask = temp['nova_attention_mask'][: max_len, : max_len]
        
            if len(input_ids) <= max_len:
                input_ids += [pad_id] * (max_len - len(input_ids))
                labels += [-100] * (max_len - len(labels))
            else:
                input_ids = input_ids[: max_len]
                labels = labels[: max_len]

            assert len(input_ids) == len(labels) == max_len
            result[f'{k}-input_ids'].append(np.array(input_ids, dtype=np.int16))
            result[f'{k}-labels'].append(np.array(labels, dtype=np.int16))
            result[f'{k}-nova_attention_mask'].append(nova_attention_mask)
    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    nova_tokenizer = NovaTokenizer(tokenizer)

    dataset = prepare_AnghaBench_contra_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/nova-anghabench-contra', max_shard_size="1GB")

    dataset = prepare_the_stack_contra_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/nova-the-stack-contra', max_shard_size="1GB")
