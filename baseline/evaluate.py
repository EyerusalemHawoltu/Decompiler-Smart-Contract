import json
import os
import numpy as np
import subprocess
import math


def re_compile(func, tmp_file):
    src = func.strip() + '\n'
    src += """
int main() {
    return 0;
}
"""
    os.chdir('/tmp/')
    with open(tmp_file, 'w') as wp:
        wp.write(src)
    try:
        subprocess.run(
            ["gcc", "-o", tmp_file.replace('.c', '.o'), tmp_file],
            check=True, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        return False
    return True


def re_execute(func, test, tmp_file):
    os.chdir('/tmp/')
    with open(tmp_file, 'w') as wp:
        wp.write(func.strip() + '\n\n')
        wp.write(test)
        
    if os.path.exists(tmp_file.replace('.c', '.o')):
        os.remove(tmp_file.replace('.c', '.o'))
    try:
        subprocess.run(
            ["gcc", "-o", tmp_file.replace('.c', '.o'), tmp_file],
            check=True, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            [f"./{tmp_file.replace('.c', '.o')}"],
            check=True, stderr=subprocess.DEVNULL, timeout=2
        )
    except Exception as e:
        return False
    return True


def run_testcases(file, wd):
    data = json.load(open(file, 'r'))
    execute_result = {'O0': [], 'O1': [], 'O2': [], 'O3': []}
    # compile_result = {'O0': [], 'O1': [], 'O2': [], 'O3': []}
    for i, item in enumerate(data):
        
        compile_correct, execute_correct = 0, 0
        for output in item['infer_c_func']:
            includes = [l for l in item['c_func'].splitlines() if l.startswith('#include')]
            includes = '\n'.join(includes)

            # compile = re_compile(includes + '\n\n' + output['c_func'], 'temp.c')
            # output['re-compile'] = compile
            # if compile:
            #     compile_correct += 1

            execute = re_execute(includes + '\n\n' + output['c_func'], item['c_test'], 'temp.c')
            output['re-execute'] = execute
            if execute:
                execute_correct += 1

        # compile_result[item['type']].append(compile_correct / len(item['infer_c_func']))
        execute_result[item['type']].append(execute_correct / len(item['infer_c_func']))
        
        print(item['task_id'], item['type'], execute_correct / len(item['infer_c_func']))
    
    os.chdir(wd)
    json.dump(data, open(file, 'w'), indent=2)


def calculate_passk(file, N=20, k=10):
    """
    N: the number of recovery sampled for each task
    k: the valud of k in Pass@k
    """
    def calculate_combinations(n, k):
        if n < k:
            return 0
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    def passk(n, c, k):
        return 1 - calculate_combinations(n - c, k) / calculate_combinations(n, k)

    result = {
        'O0-execute': [], 'O1-execute': [], 'O2-execute': [], 'O3-execute': []
    }
    data = json.load(open(file, 'r'))
    for i, item in enumerate(data):
        # compile = [output['re-compile'] for output in item['infer_c_func'][:N]]
        execute = [output['re-execute'] for output in item['infer_c_func'][:N]]
        
        # compile_cnt = compile.count(True)
        # compile = passk(N, compile_cnt, k)
        execute_cnt = execute.count(True)
        execute = passk(N, execute_cnt, k)

        result[f'{item["type"]}-execute'].append(execute)

    print('=======================================')
    for opt in result:
        print(f'Pass@{k}:', opt, np.mean(result[opt]))


if __name__ == '__main__':
    wd = os.getcwd()

    run_testcases('../benchmark/humaneval_decompile_baseline_1.3b.json', wd)
    calculate_passk('../benchmark/humaneval_decompile_baseline_1.3b.json', N=20, k=1)
    calculate_passk('../benchmark/humaneval_decompile_baseline_1.3b.json', N=20, k=10)
