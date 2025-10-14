import argparse
import random
import re
from fractions import Fraction
from typing import List, Set


# -----------------------------
# 分数格式化：支持带分数与真分数
# -----------------------------
def render_frac(val: Fraction) -> str:
    nume, deno = val.numerator, val.denominator
    if deno == 1:
        return str(nume)
    if abs(nume) < deno:
        return f"{abs(nume)}/{deno}" if val >= 0 else f"-{abs(nume)}/{deno}"
    whole_part = nume // deno
    remainder = abs(nume % deno)
    prefix = '-' if val < 0 else ''
    return f"{prefix}{abs(whole_part)}'{remainder}/{deno}"


# -----------------------------
# 表达式节点类
# -----------------------------
class CalcNode:
    def __init__(self, const=None, lhs=None, oper=None, rhs=None):
        self.const = const
        self.lhs = lhs
        self.oper = oper
        self.rhs = rhs

    def compute(self) -> Fraction:
        if self.const is not None:
            return self.const
        left_val = self.lhs.compute()
        right_val = self.rhs.compute()
        if self.oper == '+': return left_val + right_val
        if self.oper == '-': return left_val - right_val
        if self.oper == '*': return left_val * right_val
        if self.oper == '/':
            if right_val == 0:
                raise ZeroDivisionError("除数为零")
            return left_val / right_val
        raise ValueError("未知运算符")

    def to_text(self) -> str:
        if self.const is not None:
            return render_frac(self.const)
        left_txt = self.lhs.to_text()
        right_txt = self.rhs.to_text()
        return f"({left_txt} {self.oper} {right_txt})"

    def unique_key(self) -> str:
        """生成用于去重的规范键"""
        if self.const is not None:
            return render_frac(self.const)
        left_key = self.lhs.unique_key()
        right_key = self.rhs.unique_key()
        if self.oper in ('+', '*'):
            # 仅对直接子节点排序，符合题目去重要求
            a, b = sorted([left_key, right_key])
            return f"({a} {self.oper} {b})"
        return f"({left_key} {self.oper} {right_key})"

    def op_count(self) -> int:
        return 0 if self.const is not None else (1 + self.lhs.op_count() + self.rhs.op_count())


# -----------------------------
# 随机数生成
# -----------------------------
def make_random_value(bound: int) -> Fraction:
    if bound <= 1:
        return Fraction(0)
    if random.random() < 0.5:
        return Fraction(random.randint(0, bound - 1))
    denom = random.randint(2, bound - 1)
    numer = random.randint(1, denom - 1)
    return Fraction(numer, denom)


# -----------------------------
# 表达式构造
# -----------------------------
def build_expression(bound: int, max_depth: int) -> CalcNode:
    if max_depth <= 0:
        return CalcNode(const=make_random_value(bound))

    if random.random() < 0.35:
        return CalcNode(const=make_random_value(bound))

    left_depth = random.randint(0, max_depth - 1)
    right_depth = max_depth - 1 - left_depth

    left_expr = build_expression(bound, left_depth)
    right_expr = build_expression(bound, right_depth)

    ops_pool = ['+', '-', '*', '/']
    for _ in range(4):
        op = random.choice(ops_pool)
        ops_pool.remove(op)
        try:
            lv = left_expr.compute()
            rv = right_expr.compute()
            if op == '-' and lv < rv:
                continue
            if op == '/':
                if rv == 0 or lv / rv >= 1 or lv / rv < 0:
                    continue
            candidate = CalcNode(lhs=left_expr, oper=op, rhs=right_expr)
            result = candidate.compute()
            if result < 0:
                continue
            return candidate
        except:
            continue
    return CalcNode(const=make_random_value(bound))


# -----------------------------
# 生成不重复题目
# -----------------------------
def produce_problems(count: int, range_limit: int) -> List[CalcNode]:
    problem_list = []
    seen_signatures: Set[str] = set()
    tries = 0
    max_tries = max(count * 60, 1200)

    while len(problem_list) < count and tries < max_tries:
        tries += 1
        expr = build_expression(range_limit, 3)
        if expr.op_count() == 0:
            continue
        try:
            value = expr.compute()
            if value < 0:
                continue
        except:
            continue
        sig = expr.unique_key()
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            problem_list.append(expr)
    return problem_list[:count]


# -----------------------------
# 解析用户答案
# -----------------------------
def decode_answer(text: str) -> Fraction:
    s = text.strip()
    if "'" in s:
        parts = s.split("'")
        whole = parts[0]
        frac_part = parts[1]
        numer, denom = frac_part.split('/')
        sign = -1 if whole.startswith('-') else 1
        w = abs(int(whole)) if whole not in ('', '-') else 0
        return sign * (w * int(denom) + int(numer)) / int(denom)
    elif '/' in s:
        n, d = s.split('/')
        return Fraction(int(n), int(d))
    else:
        return Fraction(int(s))


# -----------------------------
# 主程序入口
# -----------------------------
def run_app():
    parser = argparse.ArgumentParser(prog="小学算术题生成器")
    parser.add_argument('-n', type=int, help='题目数量')
    parser.add_argument('-r', type=int, help='数值上限（不包含）')
    parser.add_argument('-e', type=str, help='题目文件')
    parser.add_argument('-a', type=str, help='答案文件')
    opts = parser.parse_args()

    if opts.e and opts.a:
        with open(opts.e, 'r', encoding='utf-8') as ef, open(opts.a, 'r', encoding='utf-8') as af:
            exercises = ef.readlines()
            answers = af.readlines()
        correct_ids = []
        wrong_ids = []
        for idx, (ex_line, ans_line) in enumerate(zip(exercises, answers), start=1):
            # 处理题目行：去除序号和等号
            ex_clean = ex_line.strip()
            if ex_clean.endswith('='):
                ex_clean = ex_clean[:-1].strip()
            if '. ' in ex_clean:
                expr_raw = ex_clean.split('. ', 1)[1]
            else:
                expr_raw = ex_clean

            # 处理答案行：去除序号
            ans_clean = ans_line.strip()
            if '. ' in ans_clean:
                ans_part = ans_clean.split('. ', 1)[1]
            else:
                ans_part = ans_clean

            try:
                expr_for_eval = expr_raw.replace('×', '*').replace('÷', '/')
                expr_for_eval = re.sub(r"(-?\d+)'(\d+)/(\d+)", r"(\1 + \2/\3)", expr_for_eval)
                expr_for_eval = re.sub(r"(\d+)/(\d+)", r"Fraction(\1,\2)", expr_for_eval)
                computed = eval(expr_for_eval, {"Fraction": Fraction})
                user_val = decode_answer(ans_part)
                if computed == user_val:
                    correct_ids.append(idx)
                else:
                    wrong_ids.append(idx)
            except:
                wrong_ids.append(idx)
        with open('Grade.txt', 'w', encoding='utf-8') as gf:
            gf.write("Correct: {} ({})\n".format(len(correct_ids), ", ".join(map(str, correct_ids))))
            gf.write("Wrong: {} ({})\n".format(len(wrong_ids), ", ".join(map(str, wrong_ids))))
        return

    if opts.n is None or opts.r is None:
        print("错误：必须同时指定 -n 和 -r 参数。")
        parser.print_usage()
        return

    if opts.r < 1:
        print("错误：-r 必须为正整数。")
        return

    quiz_set = produce_problems(opts.n, opts.r)
    with open('Exercises.txt', 'w', encoding='utf-8') as exf, open('Answers.txt', 'w', encoding='utf-8') as anf:
        for idx, item in enumerate(quiz_set, start=1):
            expr_text = item.to_text()
            if expr_text.startswith('(') and expr_text.endswith(')'):
                expr_text = expr_text[1:-1]
            exf.write(f"{idx}. {expr_text} =\n")
            ans_val = item.compute()
            anf.write(f"{idx}. {render_frac(ans_val)}\n")


if __name__ == '__main__':
    run_app()