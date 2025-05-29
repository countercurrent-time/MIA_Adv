import argparse
import json
import os
import re
import random
import secrets
import string

# ---------- Utility Functions ----------

def generate_random_constant():
    """生成一个随机常量：要么数字，要么字符串。"""
    if random.choice([True, False]):
        return str(random.randint(0, 9999))
    else:
        char_pool = string.ascii_letters + string.digits + string.punctuation
        rng = secrets.SystemRandom()
        length = 12
        return f"'{''.join(rng.choice(char_pool) for _ in range(length))}'"


def find_variables(code):
    """正则匹配 Python 变量名。"""
    pattern = r"^(?!(False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)$)(?!^[A-Z])[_a-z][_a-zA-Z0-9]*$"
    tokens = re.findall(r"\b[_a-zA-Z][_a-zA-Z0-9]*\b", code)
    return [tok for tok in set(tokens) if re.match(pattern, tok)]


def find_methods(code):
    """正则匹配 Python 方法名。"""
    method_pattern = re.compile(r'''^\s*def\s+
        (?P<name>(?!
            (False|None|True|and|as|assert|async|await|break|class|continue|
            def|del|elif|else|except|finally|for|from|global|if|import|in|is|
            lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b
        )(?!^[A-Z])[_a-zA-Z][_a-zA-Z0-9]*)''', re.VERBOSE | re.MULTILINE)
    return [m.group('name') for m in method_pattern.finditer(code)]

# ---------- Perturbation Variants ----------

def insert_false_if_fixed(code):
    """插入定长的无用 if 语句片段。"""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    cond = random.choice(["if \"key\" != \"key\":", "if False:"])
    body = random.choice(["    void_array = [''] * 50", "    void_array[10] = 'A'"])
    new_lines = lines[:idx] + [cond] + [body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_false_if_random(code):
    """从原代码随机摘取一行做为 if 体。"""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    cond = random.choice(["if \"key\" != \"key\":", "if False:"])
    candidates = [l.strip() for l in lines if l.strip() and not l.strip().endswith(':')]
    stmt = random.choice(candidates) if candidates else 'pass'
    body = f"    {stmt}"
    new_lines = lines[:idx] + [cond] + [body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_unused_var_existing(code):
    """插入使用已有变量初始化的未使用变量声明。"""
    vars = find_variables(code)
    val = random.choice(vars) if vars else generate_random_constant()
    new_var = f"unused_{random.randint(1000,9999)}"
    decl = f"{new_var} = {val}"
    # return decl + '\n' + code

    lines = code.splitlines()
    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{val}\b", l)), None)
    new_lines = lines[:idx] + [decl] + lines[idx:]
    return '\n'.join(new_lines)



def insert_unused_var_random(code):
    """插入使用随机常量初始化的未使用变量声明。"""
    val = generate_random_constant()
    new_var = f"unused_{random.randint(1000,9999)}"
    decl = f"{new_var} = {val}"
    # return decl + '\n' + code

    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    new_lines = lines[:idx] + [decl] + lines[idx:]
    return '\n'.join(new_lines)



def rename_var(code):
    """随机重命名一个变量或方法。"""
    vars = find_variables(code)
    if vars == []:
        return code
    name = random.choice(vars)
    prefix = 'var_'
    new_name = prefix + str(random.randint(1000,9999))
    return re.sub(rf"\b{name}\b", new_name, code)


def rename_method(code):
    """随机重命名一个变量或方法。"""
    methods = find_methods(code)
    if methods == []:
        return code
    name = random.choice(methods)
    prefix = 'method_'
    new_name = prefix + str(random.randint(1000,9999))
    return re.sub(rf"\b{name}\b", new_name, code)


def insert_print_enter(code):
    """在方法体开始处插入 Debug 进入方法的打印。"""
    lines = code.splitlines()
    methods = find_methods(code)
    name = random.choice(methods) if methods else 'foo'
    stmt = f'print("Debug: Entering method {name}()")'
    # 插入到首行
    # return stmt + '\n' + code

    # 插入到方法定义处的下一行：
    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{name}\b", l)), None)

    if idx is None:
        # 如果根本没找到，就插到文件开头
        idx = 0
        indent = ''
    else:
        # 根据那一行计算缩进
        matches = re.match(r"^\s*", lines[idx])
        if matches == None:
            return code
        else:
            indent = matches.group(0)
        # indent = re.match(r"^\\s*", lines[idx]).group(0)

    # 插在该行之后
    lines.insert(idx + 1, indent + stmt)
    return "\n".join(lines)



# def insert_print_variable(code):
#     """在变量定义处后插入 Debug 打印变量的值。"""
#     lines = code.splitlines()
#     vars = find_variables(code)
#     name = random.choice(vars) if vars else 'x'
#     stmt = f'print("Debug: Variable {name} = ", {name})'
#     # 找到变量行
#     idx = next((i for i,l in enumerate(lines) if re.search(rf"\b{name}\b", l)), len(lines)-1)
#     indent = re.match(r"^\s*", lines[idx]).group(0)
#     lines.insert(idx+1, indent + stmt)
#     return '\n'.join(lines)

def insert_print_variable(code):
    """在变量定义处后插入 Debug 打印变量的值。"""
    lines = code.splitlines()
    # 如果原代码没有任何行，直接返回
    if not lines:
        return code

    vars = find_variables(code)
    name = random.choice(vars) if vars else 'x'
    stmt = f'print("Debug: Variable {name} = ", {name})'

    # 找到第一个包含变量名的行
    # 注意把 default 换成 None，这样不会直接得出 len(lines)-1
    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{name}\b", l)), None)

    if idx is None:
        # 如果根本没找到，就插到文件开头
        idx = 0
        indent = ''
    else:
        # 根据那一行计算缩进
        matches = re.match(r"^\s*", lines[idx])
        if matches == None:
            return code
        else:
            indent = matches.group(0)
        # indent = re.match(r"^\\s*", lines[idx]).group(0)

    # 插在该行之后
    lines.insert(idx + 1, indent + stmt)
    return "\n".join(lines)


def insert_false_loop_for(code):
    """插入一个永远不执行的 for 循环。"""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    x, y = sorted([random.randint(0,9999) for _ in range(2)], reverse=True)
    loop = f"for _ in range({x}, {y}):"
    body = "    print(\"Debug: Entering loop\")" if random.choice([True, False]) else "    pass"
    new_lines = lines[:idx] + [loop, body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_false_loop_while(code):
    """插入一个永远不执行的 while 循环。"""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    cond = random.choice(["while False:", "while \"key\" != \"key\":"])
    body = "    print(\"Debug: Entering loop\")" if random.choice([True, False]) else "    pass"
    new_lines = lines[:idx] + [cond, body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_unused_var_loop(code):
    """组合：先插入无用循环，再插入未使用变量声明。"""
    # return insert_unused_var_existing(insert_false_loop_for(code))
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    x, y = sorted([random.randint(0,9999) for _ in range(2)], reverse=True)

    loop = f"for _ in range({x}, {y}):"
    val = generate_random_constant()
    new_var = f"unused_{random.randint(1000,9999)}"
    body = f"{new_var} = {val}"
    new_lines = lines[:idx] + [loop, body] + lines[idx:]
    return '\n'.join(new_lines)


PERTURBATIONS = {
    'false_if_fixed': insert_false_if_fixed,
    'false_if_random': insert_false_if_random,
    'unused_var_existing': insert_unused_var_existing,
    'unused_var_random': insert_unused_var_random,
    'rename_var': rename_var,
    'rename_method': rename_method,
    'print_enter': insert_print_enter,
    'print_variable': insert_print_variable,
    'false_loop_for': insert_false_loop_for,
    'false_loop_while': insert_false_loop_while,
    'unused_var_loop': insert_unused_var_loop
}

VARIANT_GROUPS = {
    'false_if': ['false_if_fixed', 'false_if_random'],
    'unused_var': ['unused_var_existing', 'unused_var_random'],
    'rename': ['rename_var', 'rename_method'],
    'print': ['print_enter', 'print_variable'],
    'false_loop': ['false_loop_for', 'false_loop_while'],
    'unused_var_loop': ['unused_var_loop']
}


def apply_perturbation(code, variant=None):
    """按指定 variant 应用扰动；如无指定，则随机选择。"""
    if variant is None:
        variant = random.choice(list(PERTURBATIONS.keys()))
    if variant not in PERTURBATIONS:
        raise ValueError(f"Unknown variant {variant}")
    return PERTURBATIONS[variant](code)


def traverse_all_variants(code):
    """遍历所有具体 variant，返回字典：{variant_name: perturbed_code}."""
    results = {}
    for name, func in PERTURBATIONS.items():
        results[name] = func(code)
    return results


def process_file(input_file, output_file):
    """
    处理 JSON 文件，为每个样本生成 num_perturbations 个扰动版本。
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = json.loads(line.strip())  # 解析每一行的 JSON 数据
            
            if line:
                original_input = line['input']
                gt = line['gt']  # 原始输出 (ground truth)

                # 构建新的样本格式
                sample = {
                    "id": str(line["id"]),
                    "input": original_input,
                    "gt": gt
                }
                # 写入文件，每行一个 JSON 对象
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

                # 生成指定数量的扰动版本
                perturbed_code = traverse_all_variants(original_input.replace('\\n', '\n').replace('<s> ', '', 1)).values()
                for perturbed_input in perturbed_code:
                    sample = {
                        "id": str(line["id"]),
                        "input": '<s> ' + perturbed_input.replace('\n', '\\n'),
                        "gt": gt
                    }
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input directory path.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The json input file contains input code and ground truth for each line.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The json output file contains perturbed input code and ground truth for each line.")


    args = parser.parse_args()
    process_file(os.path.join(args.input_dir, args.input_file), os.path.join(args.input_dir, args.output_file))
    print(f"处理完成！结果已保存到 {args.output_file}")


if __name__ == '__main__':
    main()
