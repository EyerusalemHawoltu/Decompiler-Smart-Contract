import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def prepare_AnghaBench_data():
    data = {'task': [], 'input': [], 'output': []}
    with open(f'../data/anghabench-normalize.jsonl', 'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)

            data['task'].append(item['name'].split('/')[-1])
            data['input'].append('')
            data['output'].append(item['input'].strip())

            for opt in ['O0', 'O1', 'O2', 'O3']:
                data['task'].append(item['name'].split('/')[-1])
                data['input'].append('')
                data['output'].append(item['output'][f'opt-state-{opt}'].strip())

    print(len(data['task']))
    
    train = {k: v[: -5000] for k, v in data.items()}
    valid = {k: v[-5000: ] for k, v in data.items()}
    
    return DatasetDict({'train': Dataset.from_dict(train), 'valid': Dataset.from_dict(valid)})


def prepare_the_stack_data():
    data = {'task': [], 'input': [], 'output': []}
    
    with open(f'../data/the-stack-normalize.jsonl', 'r') as fp:
        L = fp.readlines()

    for line in L:
        item = json.loads(line)

        data['task'].append(item['name'].split('/')[-1])
        data['input'].append('')
        data['output'].append(item['input'].strip())

        for opt in ['O0', 'O1', 'O2', 'O3']:
            if len(item['output']['opt-state-O0'].splitlines()) >= 10:
                data['task'].append(item['name'].split('/')[-1])
                data['input'].append('')
                data['output'].append(item['output'][f'opt-state-{opt}'].strip())

    print(len(data['task']))
    
    train = {k: v[: -1000] for k, v in data.items()}
    valid = {k: v[-1000: ] for k, v in data.items()}
    
    return DatasetDict({'train': Dataset.from_dict(train), 'valid': Dataset.from_dict(valid)})


def tokenize_dataset(batch):
    global tokenizer
    max_len = 2048
    
    pad_id = tokenizer.eos_token_id
    result = {'input_ids': [], 'labels': [], 'input': [], 'output': []}
    for i in range(len(batch['input'])):
        inputs = tokenizer.encode(batch['input'][i])[1: ]
        outputs = tokenizer.encode(batch['output'][i])[1: ]
        
        input_ids = inputs + outputs
        labels = [-100] * len(inputs) + outputs
        
        if len(input_ids) <= max_len:
            input_ids += [pad_id] * (max_len - len(input_ids))
            labels += [-100] * (max_len - len(labels))
        else:
            input_ids = input_ids[: max_len]
            labels = labels[: max_len]

        assert len(input_ids) == len(labels) == max_len
        result['input'].append(batch['input'][i])
        result['output'].append(batch['output'][i])
        result['input_ids'].append(input_ids)
        result['labels'].append(labels)
    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    print(len(tokenizer.get_vocab()), len(tokenizer.get_vocab()) / 4)

    dataset = prepare_the_stack_data()
    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset.save_to_disk('../data-tokenized/the-stack-lm/', max_shard_size="1GB")

    dataset = prepare_AnghaBench_data()
    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset.save_to_disk('../data-tokenized/anghabench-lm/', max_shard_size="1GB")
