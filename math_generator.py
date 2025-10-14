import argparse
import random
import math
import re
import os
from fractions import Fraction
from itertools import combinations
from typing import List, Tuple, Set


# ========================
# 自定义分数格式化函数
# ========================
def format_fraction(frac: Fraction) -> str:
    if frac.denominator == 1:
        return str(frac.numerator)
    elif abs(frac.numerator) < frac.denominator:
        return f"{abs(frac.numerator)}/{frac.denominator}" if frac >= 0 else f"-{abs(frac.numerator)}/{frac.denominator}"
    else:
        whole = frac.numerator // frac.denominator
        remainder = abs(frac.numerator % frac.denominator)
        if remainder == 0:
            return str(whole)
        sign = '-' if frac < 0 else ''
        return f"{sign}{abs(whole)}'{remainder}/{frac.denominator}"


# ========================
# 表达式类（用于去重）
# ========================
class Expr:
    def __init__(self, value=None, left=None, op=None, right=None):
        self.value = value  # 用于叶子节点（数字）
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self) -> Fraction:
        if self.value is not None:
            return self.value
        l = self.left.evaluate()
        r = self.right.evaluate()
        if self.op == '+':
            return l + r
        elif self.op == '-':
            return l - r
        elif self.op == '*':
            return l * r
        elif self.op == '/':
            if r == 0:
                raise ZeroDivisionError
            return l / r

    def to_string(self) -> str:
        if self.value is not None:
            return format_fraction(self.value)
        left_str = self.left.to_string()
        right_str = self.right.to_string()
        # 加括号规则：仅当子表达式优先级低于当前操作符时才加括号
        # 为简化，这里统一加括号（避免歧义）
        return f"({left_str} {self.op} {right_str})"

    def canonical_form(self) -> str:
        """生成规范表达式用于去重"""
        if self.value is not None:
            return format_fraction(self.value)
        l = self.left.canonical_form()
        r = self.right.canonical_form()
        # 对交换律操作符排序
        if self.op in ('+', '*'):
            parts = sorted([l, r])
            return f"({parts[0]} {self.op} {parts[1]})"
        else:
            return f"({l} {self.op} {r})"

    def count_ops(self) -> int:
        if self.value is not None:
            return 0
        return 1 + self.left.count_ops() + self.right.count_ops()


# ========================
# 随机生成表达式
# ========================
def generate_number(r: int) -> Fraction:
    """生成一个 [0, r) 范围内的自然数或真分数"""
    if random.choice([True, False]):
        # 自然数
        return Fraction(random.randint(0, r - 1))
    else:
        # 真分数：分子 < 分母，分母 ∈ [2, r)
        if r <= 2:
            return Fraction(0)
        denom = random.randint(2, r - 1)
        numer = random.randint(1, denom - 1)
        return Fraction(numer, denom)


def generate_expr(r: int, max_ops: int = 3) -> Expr:
    """递归生成合法表达式，最多 max_ops 个运算符"""
    if max_ops == 0:
        return Expr(value=generate_number(r))

    # 随机决定是否继续组合
    if random.random() < 0.4 or max_ops == 1:
        return Expr(value=generate_number(r))

    left_ops = random.randint(0, max_ops - 1)
    right_ops = max_ops - 1 - left_ops

    left = generate_expr(r, left_ops)
    right = generate_expr(r, right_ops)

    ops = ['+', '-', '*', '/']
    random.shuffle(ops)
    for op in ops:
        try:
            l_val = left.evaluate()
            r_val = right.evaluate()
            if op == '-':
                if l_val < r_val:
                    continue
            if op == '/':
                if r_val == 0 or l_val / r_val >= 1 or l_val / r_val < 0:
                    continue
            expr = Expr(left=left, op=op, right=right)
            # 验证整个表达式合法性（无负数、除法结果为真分数）
            val = expr.evaluate()
            if val < 0:
                continue
            # 除法子表达式检查已在上层处理，这里只检查整体
            return expr
        except:
            continue
    # 如果所有操作都不合法，退化为单个数字
    return Expr(value=generate_number(r))


# ========================
# 去重生成题目
# ========================
def generate_exercises(n: int, r: int) -> List[Expr]:
    exercises: List[Expr] = []
    seen: Set[str] = set()
    attempts = 0
    max_attempts = n * 20  # 防止无限循环

    while len(exercises) < n and attempts < max_attempts:
        attempts += 1
        expr = generate_expr(r, max_ops=3)
        if expr.count_ops() == 0:
            continue
        try:
            val = expr.evaluate()
            if val < 0:
                continue
        except:
            continue
        canon = expr.canonical_form()
        if canon not in seen:
            seen.add(canon)
            exercises.append(expr)
    if len(exercises) < n:
        print(f"Warning: Only generated {len(exercises)} unique exercises.")
    return exercises[:n]


# ========================
# 解析答案字符串为 Fraction
# ========================
def parse_answer(ans_str: str) -> Fraction:
    ans_str = ans_str.strip()
    if "'" in ans_str:
        whole_part, frac_part = ans_str.split("'")
        sign = -1 if whole_part.startswith('-') else 1
        whole = abs(int(whole_part)) if whole_part not in ('', '-') else 0
        numer, denom = map(int, frac_part.split('/'))
        return sign * (whole * denom + numer) / denom
    elif '/' in ans_str:
        numer, denom = map(int, ans_str.split('/'))
        return Fraction(numer, denom)
    else:
        return Fraction(int(ans_str))


# ========================
# 主函数
# ========================
def main():
    parser = argparse.ArgumentParser(description="小学四则运算题目生成器")
    parser.add_argument('-n', type=int, help='生成题目数量')
    parser.add_argument('-r', type=int, help='数值范围（不包括该数）')
    parser.add_argument('-e', type=str, help='题目文件路径')
    parser.add_argument('-a', type=str, help='答案文件路径')
    args = parser.parse_args()

    if args.e and args.a:
        # 校对模式
        with open(args.e, 'r', encoding='utf-8') as ef, open(args.a, 'r', encoding='utf-8') as af:
            exercises = ef.readlines()
            answers = af.readlines()
        correct = []
        wrong = []
        for i, (ex, ans) in enumerate(zip(exercises, answers), 1):
            # 提取表达式部分（去掉末尾的 =）
            expr_str = ex.strip().rstrip(' =')
            try:
                # 用 eval 不安全，但题目格式固定，且为教学用途
                # 更安全做法是自己解析，但为简化使用 Fraction 和替换
                expr_eval = expr_str.replace('×', '*').replace('÷', '/')

                # 处理带分数：如 2'3/4 → (2 + 3/4)
                def replace_mixed(m):
                    whole = m.group(1)
                    numer = m.group(2)
                    denom = m.group(3)
                    return f"({whole} + {numer}/{denom})"

                expr_eval = re.sub(r"(-?\d+)'(\d+)/(\d+)", replace_mixed, expr_eval)
                # 替换普通分数
                expr_eval = re.sub(r"(\d+)/(\d+)", r"Fraction(\1, \2)", expr_eval)
                # 添加 Fraction 导入
                val = eval(expr_eval, {"Fraction": Fraction, "__builtins__": {}})
                user_ans = parse_answer(ans)
                if val == user_ans:
                    correct.append(i)
                else:
                    wrong.append(i)
            except Exception as e:
                wrong.append(i)
        with open('Grade.txt', 'w', encoding='utf-8') as gf:
            gf.write(f"Correct: {len(correct)} ({', '.join(map(str, correct))})\n")
            gf.write(f"Wrong: {len(wrong)} ({', '.join(map(str, wrong))})\n")
        return

    if args.n is None or args.r is None:
        parser.print_help()
        print("\n错误：必须同时提供 -n 和 -r 参数。")
        return

    if args.r <= 0:
        print("错误：-r 参数必须是正整数。")
        return

    exercises = generate_exercises(args.n, args.r)
    with open('Exercises.txt', 'w', encoding='utf-8') as ef, open('Answers.txt', 'w', encoding='utf-8') as af:
        for i, expr in enumerate(exercises, 1):
            expr_str = expr.to_string()
            # 去掉最外层括号（如果存在）
            if expr_str.startswith('(') and expr_str.endswith(')'):
                expr_str = expr_str[1:-1]
            ef.write(f"{expr_str} =\n")
            ans = expr.evaluate()
            af.write(f"{format_fraction(ans)}\n")


if __name__ == '__main__':
    main()