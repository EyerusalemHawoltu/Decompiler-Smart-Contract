import re
import os
import json
import argparse
import subprocess
from datasets import load_dataset

from utils import remove_c_comments, find_functions, filter_self_contained_func


def download_the_stack(max_count=200):
    ds = load_dataset("bigcode/the-stack-dedup", streaming=True, split="train", data_dir="data/c")
    cnt = 0
    for idx, sample in enumerate(iter(ds)):
        assert sample['lang'] == 'C'
        
        with open(f'the-stack/c/{idx}.c', 'w') as wp:
            wp.write(sample['content'])

        cnt += 1
        # download 200 files for a simple test
        if cnt >= max_count:
            break


def compile_the_stack(output_file):
    result = []
    OPT = ["O0", "O1", "O2", "O3"]
    L = os.listdir('the-stack/c/')
    for i, file in enumerate(L):
        if i % 100 == 0:
            print(f"{i}/{len(L)}, {len(result)}")

        if not file.endswith('.c'):
            continue
        
        src = ''.join(open('the-stack/c/' + file, 'r').readlines())
        src = remove_c_comments(src)
        
        try: 
            function_list = find_functions('the-stack/c/' + file)
        except Exception as e:
            print(e)
            continue
        
        temp = {}
        for opt in OPT:
            try:
                subprocess.run(
                    ["gcc", "-c", "-o", f"the-stack/bin/{file.split('.')[0]}-{opt}.o", 
                    'the-stack/c/' + file, "-" + opt],
                    check=True, stderr=subprocess.DEVNULL
                )
            except:
                continue
        
            for func in function_list:
                src = remove_c_comments(function_list[func])
            
                if func not in temp:
                    temp[func] = {'src': src}
                
                asm = subprocess.check_output([f"objdump -d the-stack/bin/{file.split('.')[0]}-{opt}.o | awk -v RS= '/^[[:xdigit:]]+? <{func}>/'"], shell=True, encoding='utf-8').strip()
                if asm.strip() == "":
                    continue
                asm_clean = ""
                
                for tmp in asm.split("\n"):
                    if len(tmp.split("\t")) == 3 and tmp.split("\t")[0].endswith(':'):
                        tmp_asm = tmp.split("\t")[0] + '\t' + tmp.split("\t")[-1]
                    else:
                        tmp_asm = tmp
                    tmp_asm = tmp_asm.split("#")[0].strip()
                    asm_clean += tmp_asm + "\n"

                asm = re.sub(r"^0{2,}", "", asm_clean, flags=re.MULTILINE)
                temp[func]["opt-state-" + opt] = asm.strip()
        for func in temp:
            if temp[func]['src'] == '' or len(temp[func]) <= 1:
                continue
            result.append({
                'name': f'{file}-{func}',
                'input': temp[func]['src'],
                'output': {'opt-state-' + opt: temp[func]['opt-state-' + opt] for opt in OPT if 'opt-state-' + opt in temp[func]}
            })
    with open(output_file, 'w') as wp:
        for line in result:
            wp.write(json.dumps(line) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile C files and generate JSONL output."
    )
    parser.add_argument("--output", default="the-stack/the-stack.jsonl", help="Path to JSONL output file.")
    parser.add_argument("--n", type=int, default=None, help="Number of files to compile")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    output_file = args.output

    download_the_stack(max_count=args.n)
    compile_the_stack(output_file=output_file)

    # optional if need less and easier data
    # filter_self_contained_func(output_file)


if __name__ == '__main__':
    main()