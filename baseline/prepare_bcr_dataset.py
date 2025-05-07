import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import clang.cindex
clang.cindex.Config.set_library_file('/usr/lib/llvm-12/lib/libclang-12.so.1')


def count_functions(file_content, file_name='temp_file.c'):
    # Create a new index for parsing
    index = clang.cindex.Index.create()
    
    # Parse the given file
    tu = index.parse(file_name, unsaved_files=[(file_name, file_content)])
    
    # Function to recursively visit nodes in the AST
    def visit_node(node, function_list):
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.is_definition():
            # If the node is a function declaration, add it to the list
            function_list.append(node.spelling)
        # Recursively visit the children of this node
        for child in node.get_children():
            visit_node(child, function_list)

    function_list = []
    visit_node(tu.cursor, function_list)

    # Return the number of functions found
    
    return function_list


def prepare_AnghaBench_data():
    data = {'task': [], 'input': [], 'output': []}

    with open(f'../data/anghabench/anghabench-normalize.jsonl', 'r') as fp:
        L = fp.readlines()
        for _, line in enumerate(L):
            if  _ % 10000 == 0:
                print(f'{_} / {len(L)}')
                
            item = json.loads(line)
            if len(item['output']) == 0:
                continue

            try:
                function_list = count_functions(item['input'])
                if len(function_list) != 1:
                    continue
            except Exception as e:
                print(e)
                continue
            
            for opt in ['O0', 'O1', 'O2', 'O3']:
                if f'opt-state-{opt}' not in item['output']:
                    continue

                # only consider data length within 10 lines to 256 lines
                if 256 >= len(item['output'][f'opt-state-{opt}'].splitlines()) >= 10:
                    data['task'].append(item['name'].split('/')[-1])

                    inputs = f'# This is the assembly code with {opt} optimization:\n' + \
                        item['output'][f'opt-state-{opt}'].strip() + '\nWhat is the source code?\n'
                    outputs = item['input'].replace(function_list[0], 'func0').strip()
            
                    data['input'].append(inputs)
                    data['output'].append(outputs)
                        
    print(len(data['task']))
    
    train = {k: v[: -5000] for k, v in data.items()}
    valid = {k: v[-5000: ] for k, v in data.items()}
    
    return DatasetDict({'train': Dataset.from_dict(train), 'valid': Dataset.from_dict(valid)})


def prepare_the_stack_data():
    data = {'task': [], 'input': [], 'output': []}
    
    with open(f'../data/the-stack/the-stack-normalize.jsonl', 'r') as fp:
        L = fp.readlines()
    
    for _, line in enumerate(L):
        if  _ % 1000 == 0:
            print(f'{_} / {len(L)}')
            
        item = json.loads(line)
        if len(item['output']) == 0:
            continue

        try:
            function_list = count_functions(item['input'])
            if len(function_list) != 1:
                continue
        except Exception as e:
            print(e)
            continue
        
        for opt in ['O0', 'O1', 'O2', 'O3']:
            if f'opt-state-{opt}' not in item['output']:
                continue
            if 256 >= len(item['output'][f'opt-state-{opt}'].splitlines()) >= 10:
                data['task'].append(item['name'].split('/')[-1])

                inputs = f'# This is the assembly code with {opt} optimization:\n' + \
                    item['output'][f'opt-state-{opt}'].strip() + '\nWhat is the source code?\n'
                outputs = item['input'].replace(function_list[0], 'func0').strip()
        
                data['input'].append(inputs)
                data['output'].append(outputs)
                    
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
        outputs = tokenizer.encode(batch['output'][i])[1: ] + [tokenizer.eos_token_id]
        
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

    dataset = prepare_AnghaBench_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/anghabench-bcr/', max_shard_size="1GB")

    dataset = prepare_the_stack_data()
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=32)
    dataset.save_to_disk('../data-tokenized/the-stack-bcr/', max_shard_size="1GB")
