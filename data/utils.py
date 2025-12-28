import re
import json
import subprocess

# Require clang installation
import clang.cindex
clang.cindex.Config.set_library_file('/usr/lib/llvm-12/lib/libclang-12.so.1')


def find_functions(filename):
    index = clang.cindex.Index.create()
    tu = index.parse(filename)
    
    functions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    def visit_node(node):
        def format_code(code):
            proc = subprocess.run(['clang-format', '-style={IndentWidth: 4, UseTab: Never, TabWidth: 4}'], 
                                input=code, text=True, capture_output=True)
            return proc.stdout
        
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            if node.is_definition():
                extent = node.extent
                functions[node.spelling] = format_code(''.join(lines[extent.start.line-1:extent.end.line]))
        for child in node.get_children():
            visit_node(child)
    visit_node(tu.cursor)
    return functions


def remove_empty_lines(code):
    return '\n'.join([
        line for line in code.splitlines() if line.strip() != ''
    ])


def remove_c_comments(code):
    # Pattern to match single-line comments
    single_line_comment_pattern = r'//.*'
    # Pattern to match multi-line comments
    multi_line_comment_pattern = r'/\*.*?\*/'
    
    # Remove single-line comments
    code_without_single_line_comments = re.sub(single_line_comment_pattern, '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    code_without_comments = re.sub(multi_line_comment_pattern, '', code_without_single_line_comments, flags=re.DOTALL)
    
    return code_without_comments


def re_compile(func, tmp_file):
    with open(tmp_file, 'w') as wp:
        wp.write(func)
    try:
        subprocess.run(
            ["gcc", "-o", tmp_file.split('.')[0] + '.o', tmp_file],
            check=True, stderr=subprocess.DEVNULL
        )
    except:
        return False
    return True


def filter_self_contained_func(file_path):
    data = []
    with open(file_path, 'r') as fp:
        L = fp.readlines()
        cnt = 0
        for i, line in enumerate(L):
            if i % 100 == 0:
                print(i, cnt, len(L))

            item = json.loads(line)
            src = item['input']

            if 'int main' not in src:
                src += """
int main() {
    return 0;
}
"""
            if not re_compile(src, '/tmp/tmp.c'):
                continue
            data.append(item)
            cnt += 1
    
    with open(file_path, 'w') as wp:
        for item in data:
            wp.write(json.dumps(item) + '\n')
