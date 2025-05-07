import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer
import numpy as np


def prepare_AnghaBench_data():
    data = {'task': [], 'input': [], 'output': [], 'char_types': []}
    with open(f'../data/anghabench-normalize.jsonl', 'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)

            if 256 >= len(item['output']['opt-state-O0'].splitlines()) >= 10 and 256 >= len(item['output']['opt-state-O1'].splitlines()) >= 10 and \
                256 >= len(item['output']['opt-state-O2'].splitlines()) >= 10 and 256 >= len(item['output']['opt-state-O3'].splitlines()) >= 10:
                data['task'].append(item['name'].split('/')[-1])
                data['input'].append('')
                data['output'].append(item['input'].strip())
                data['char_types'].append('0' * len(item['input'].strip()))

                for opt in ['O0', 'O1', 'O2', 'O3']:
                    data['task'].append(item['name'].split('/')[-1])
                    data['input'].append('')
                    asm = item['output'][f'opt-state-{opt}'].strip()
                    assert asm.startswith('<func0>:')
                    data['output'].append(asm)
                    data['char_types'].append('0' * len('<func0>:') + '1' * (len(asm) - len('<func0>:')))

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

        if 256 >= len(item['output']['opt-state-O0'].splitlines()) >= 10 and 256 >= len(item['output']['opt-state-O1'].splitlines()) >= 10 and \
            256 >= len(item['output']['opt-state-O2'].splitlines()) >= 10 and 256 >= len(item['output']['opt-state-O3'].splitlines()) >= 10:
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
    global nova_tokenizer
    max_len = 2048
    
    pad_id = tokenizer.eos_token_id
    result = {'input_ids': [], 'labels': [], 'nova_attention_mask': [], 'input': [], 'output': []}
    for i in range(len(batch['input'])):
        assert len(batch['input'][i] + batch['output'][i]) == len(batch['char_types'][i])
        
        temp = nova_tokenizer.encode(batch['input'][i], batch['output'][i], batch['char_types'][i])
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
        result['input'].append(batch['input'][i])
        result['output'].append(batch['output'][i])
        result['input_ids'].append(np.array(input_ids, dtype=np.int16))
        result['labels'].append(np.array(labels, dtype=np.int16))
        result['nova_attention_mask'].append(nova_attention_mask)
    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
    nova_tokenizer = NovaTokenizer(tokenizer)

    dataset = prepare_the_stack_data()
    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset.save_to_disk('../data-tokenized/nova-the-stack-lm/', max_shard_size="1GB")

    dataset = prepare_AnghaBench_data()
    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset.save_to_disk('../data-tokenized/nova-anghabench-lm/', max_shard_size="1GB")
