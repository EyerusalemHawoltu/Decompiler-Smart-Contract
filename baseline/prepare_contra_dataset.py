import json
import copy
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def prepare_AnghaBench_contra_data():
    data = {'task': [], 'src': [], 'asm-O0': [], 'asm-O1': [], 'asm-O2': [], 'asm-O3': []}
    
    with open(f'../data/anghabench/anghabench-normalize.jsonl', 'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)

            # we need functions that have all four optimized assembly
            if 'opt-state-O0' not in item['output'] or 'opt-state-O1' not in item['output'] or 'opt-state-O2' not in item['output'] or 'opt-state-O3' not in item['output']:
                continue
            # there should be some difference between differently optimized assembly (optional quality check)
            if item['output']['opt-state-O0'] == item['output']['opt-state-O1'] or item['output']['opt-state-O1'] == item['output']['opt-state-O2'] or item['output']['opt-state-O2'] == item['output']['opt-state-O3']:
                continue
            data['task'].append(item['name'].split('/')[-1])
            data['src'].append(item['input'].strip())
            data['asm-O0'].append(item['output']['opt-state-O0'].strip())
            data['asm-O1'].append(item['output']['opt-state-O1'].strip())
            data['asm-O2'].append(item['output']['opt-state-O2'].strip())
            data['asm-O3'].append(item['output']['opt-state-O3'].strip())
    print(len(data['task']))
    return DatasetDict({'train': Dataset.from_dict(data)})


def prepare_the_stack_contra_data():
    data = {'task': [], 'src': [], 'asm-O0': [], 'asm-O1': [], 'asm-O2': [], 'asm-O3': []}
    
    with open(f'../data/the-stack/the-stack-normalize.jsonl', 'r') as fp:
        L = fp.readlines()
    
    for line in L:
        item = json.loads(line)

        if 'opt-state-O0' not in item['output'] or 'opt-state-O1' not in item['output'] or 'opt-state-O2' not in item['output'] or 'opt-state-O3' not in item['output']:
            continue
        if item['output']['opt-state-O0'] == item['output']['opt-state-O1'] or item['output']['opt-state-O1'] == item['output']['opt-state-O2'] or item['output']['opt-state-O2'] == item['output']['opt-state-O3']:
            continue
        data['task'].append(item['name'].split('/')[-1])
        data['src'].append(item['input'].strip())
        data['asm-O0'].append(item['output']['opt-state-O0'].strip())
        data['asm-O1'].append(item['output']['opt-state-O1'].strip())
        data['asm-O2'].append(item['output']['opt-state-O2'].strip())
        data['asm-O3'].append(item['output']['opt-state-O3'].strip())
    print(len(data['task']))
    return DatasetDict({'train': Dataset.from_dict(data)})


def tokenize_dataset(batch):
    global tokenizer
    max_len = 2048
    
    pad_id = tokenizer.eos_token_id
    result = {'src-input_ids': [], 'asm-O0-input_ids': [], 'asm-O1-input_ids': [], 'asm-O2-input_ids': [], 'asm-O3-input_ids': [],
              'src-labels': [], 'asm-O0-labels': [], 'asm-O1-labels': [], 'asm-O2-labels': [], 'asm-O3-labels': []}
    for i in range(len(batch['src'])):
        for k in batch:
            if k == 'task':
                continue
            input_ids = tokenizer.encode(batch[k][i])[1: ]
            labels = copy.deepcopy(input_ids)
        
            if len(input_ids) <= max_len:
                input_ids += [pad_id] * (max_len - len(input_ids))
                labels += [-100] * (max_len - len(labels))
            else:
                input_ids = input_ids[: max_len]
                labels = labels[: max_len]

            assert len(input_ids) == len(labels) == max_len
            result[k + '-input_ids'].append(input_ids)
            result[k + '-labels'].append(labels)
    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    print(len(tokenizer.get_vocab()), len(tokenizer.get_vocab()) / 4)

    dataset = prepare_the_stack_contra_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/the-stack-contra/', max_shard_size="1GB")

    dataset = prepare_AnghaBench_contra_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/anghabench-contra', max_shard_size="1GB")
