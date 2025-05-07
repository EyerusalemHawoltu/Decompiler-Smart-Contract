import json
import re
import os


def hex_to_decimal(matched):
    return str(int(matched.group(), 16))


def normalize(asm):
    asm = asm.strip().split('\n')[: 257]

    asm_lst = []
    addr2label = {}
    func_cnt, label_cnt = 0, 0
    for i, line in enumerate(asm):
        if line.strip() == '' or 'file format elf64-x86-64' in line:
            continue
        
        if len(line.split('\t')) == 1 and line.endswith(':'):
            func = line[line.index('<') + 1 : line.index('>')]
            asm_lst.append([f'<func{func_cnt}>:'])
            func_cnt += 1
        else:
            if len(line.split('\t')) < 2:
                print(line)
            label_cnt += 1
            addr, content = line.split('\t', 1)

            addr = addr[: -1]
            addr2label[addr] = f'<label-{label_cnt}>'
            asm_lst.append(
                [content.strip(), f'<label-{label_cnt}>']
            )
    
    new_asm = ''
    for i, item in enumerate(asm_lst):
        if len(item) == 1:
            new_asm += '\n' + item[0]
            continue
        content, label = item

        if '<' in content and '>' in content:
            content = content[: content.index('<')].strip()

        if content.startswith('j') or content.startswith('loop') or content.startswith('call'):
            if len(content.split()) == 2:
                inst, addr = content.split()
                if addr.startswith('0x'):
                    addr = addr[2:]
                if addr not in addr2label:
                    content = inst + '\t' + '<unk>'
                else:
                    content = inst + '\t' + addr2label[addr]
        content = re.sub(r"0x([0-9A-Fa-f]+)", hex_to_decimal, content)
        content = content.replace('%', '')
        content = re.sub(r"([,(])|([),])", r' \1\2 ', content)
        content = re.sub(r' +', ' ', content).strip()

        new_asm += '\n' + content + '\t' + label
    return new_asm


def normalize_anghabench():
    wp = open(f'anghabench/anghabench-normalize.jsonl', 'w')
    fail = 0
    with open(f'anghabench/anghabench.jsonl', 'r') as fp:
        L = fp.readlines()
        for i, line in enumerate(L):
            try:
                item = json.loads(line)
                for opt in item['output']:
                    item['output'][opt] = normalize(item['output'][opt])
            except Exception as e:
                fail += 1
                continue
            wp.write(json.dumps(item) + '\n')

            if i % 1000 == 0:
                print(f"{i}/{len(L)}, fail: {fail}")


def normalize_the_stack():
    wp = open('the-stack/the-stack-normalize.jsonl', 'w')
    fail = 0
    with open('the-stack/the-stack.jsonl', 'r') as fp:
        L = fp.readlines()
        for i, line in enumerate(L):
            if i % 1000 == 0:
                print(f"{i}/{len(L)}, fail: {fail}")
            try:
                item = json.loads(line)
                for opt in item['output']:
                    item['output'][opt] = normalize(item['output'][opt]).strip()
            except Exception as e:
                fail += 1
                print(e)
                continue
            wp.write(json.dumps(item) + '\n')


def normalize_codeart():
    for file in os.listdir('codeart/'):
        L = []
        with open(f'codeart/{file}', 'r') as fp:
            for l in fp.readlines():
                item = json.loads(l.strip())
                item['normalized_asm'] = normalize(item['asm'])
                L.append(item)
        
        with open(f'codeart/{file}', 'w') as wp:
            for l in L:
                wp.write(json.dumps(l) + '\n')


def normalize_binarycorp(binary_corp_folder):
    data = {}
    for file in os.listdir(binary_corp_folder):
        if '-O0-' in file:
            proj = file[: file.index('-O0-')]
            opt = 'O0'
        elif '-O1-' in file:
            proj = file[: file.index('-O1-')]
            opt = 'O1'
        elif '-O3-' in file:
            proj = file[: file.index('-O3-')]
            opt = 'O3'
        else:
            continue
        if proj not in data:
            data[proj] = {}
        content = json.load(open(f'{binary_corp_folder}/{file}', 'r'))
        for k, v in content.items():
            func = v['name']
            asm = v['assembly']
            if func not in data[proj]:
                data[proj][func] = {}
            data[proj][func][opt] = normalize(asm)
        print(len(data))
    
    data_filter = {}
    for proj in data:
        data_filter[proj] = {}
        for func in data[proj]:
            if len(data[proj][func]) < 2 or 'O3' not in data[proj][func]:
                continue
            data_filter[proj][func] = data[proj][func]
        if len(data_filter[proj]) == 0:
            data_filter.pop(proj)
    json.dump(data_filter, open('binarycorp/binarycorp.json', 'w'), indent=2)


if __name__ == '__main__':
    # training data
    normalize_the_stack()
    normalize_anghabench()

    # fine-tuning data
    # download BinaryCorp small_train.tar from https://cloud.vul337.team:8443/s/cxnH8DfZTADLKCs
    # binary_corp_folder = ''
    # normalize_binarycorp(binary_corp_folder)

    # evaluation data
    # normalize_codeart()