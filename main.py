"""
小学四则运算题目生成器
该模块用于生成符合小学教学要求的四则运算题目，支持自然数、真分数和带分数运算，
"""
import argparse
import random
import re
from fractions import Fraction
from typing import List, Set, Tuple


# -----------------------------
# 分数格式化：支持带分数与真分数
# -----------------------------
def render_frac(val: Fraction) -> str:
    """
       将Fraction对象格式化为字符串表示

       Args:
           val: 要格式化的分数

       Returns:
           格式化后的字符串，可能是整数、真分数或带分数形式
       """
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
    """表示算术表达式的节点类"""
    def __init__(self, const=None, lhs=None, oper=None, rhs=None):
        """
               初始化表达式节点

               Args:
                   const: 常量值（如果是叶子节点）
                   lhs: 左子表达式
                   oper: 运算符
                   rhs: 右子表达式
        """
        self.const = const
        self.lhs = lhs
        self.oper = oper
        self.rhs = rhs

    def compute(self) -> Fraction:
        """计算表达式的值"""
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
        """将表达式转换为文本形式"""
        if self.const is not None:
            return render_frac(self.const)

        left_txt = self.lhs.to_text()
        right_txt = self.rhs.to_text()

        # 根据运算符优先级决定是否加括号
        need_left_paren = self._need_parentheses(self.lhs, is_left=True)
        need_right_paren = self._need_parentheses(self.rhs, is_left=False)

        left_final = f"({left_txt})" if need_left_paren else left_txt
        right_final = f"({right_txt})" if need_right_paren else right_txt

        return f"{left_final} {self.oper} {right_final}"

    def _need_parentheses(self, child, is_left: bool) -> bool:
        if child.const is not None:
            return False

        current_priority = self._get_priority(self.oper)
        child_priority = self._get_priority(child.oper)

        if current_priority > child_priority:
            return True
        elif current_priority == child_priority:
            # 对于相同优先级的左结合运算符，右操作数需要括号
            if not is_left and self.oper in ['-', '/']:
                return True
        return False

    def _get_priority(self, op):
        if op in ['+', '-']:
            return 1
        if op in ['*', '/']:
            return 2
        return 0

    def unique_key(self) -> str:
        """生成用于去重的规范键"""
        if self.const is not None:
            return render_frac(self.const)

        left_key = self.lhs.unique_key()
        right_key = self.rhs.unique_key()

        if self.oper in ('+', '*'):
            # 对操作数排序
            a, b = sorted([left_key, right_key])
            return f"{a}{self.oper}{b}"
        return f"{left_key}{self.oper}{right_key}"

    def op_count(self) -> int:
        return 0 if self.const is not None else (1 + self.lhs.op_count() + self.rhs.op_count())

    def validate_subtraction(self) -> bool:
        """验证所有减法操作 e1 - e2 满足 e1 ≥ e2"""
        if self.const is not None:
            return True

        # 验证当前节点的减法
        if self.oper == '-':
            left_val = self.lhs.compute()
            right_val = self.rhs.compute()
            if left_val < right_val:
                return False

        # 递归验证子节点
        return self.lhs.validate_subtraction() and self.rhs.validate_subtraction()

    def validate_division(self) -> bool:
        """验证所有除法操作 e1 ÷ e2 的结果是真分数"""
        if self.const is not None:
            return True

        # 验证当前节点的除法
        if self.oper == '/':
            left_val = self.lhs.compute()
            right_val = self.rhs.compute()
            if right_val == 0:
                return False
            result = left_val / right_val
            if result <= 0 or result >= 1:
                return False

        # 递归验证子节点
        return self.lhs.validate_division() and self.rhs.validate_division()


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
# 表达式构造 - 简化版本
# -----------------------------
def build_expression(bound: int, max_depth: int, force_operator=False) -> CalcNode:
    if max_depth <= 0 or (not force_operator and random.random() < 0.3):
        return CalcNode(const=make_random_value(bound))

    # 选择运算符
    operators = ['+', '-', '*', '/']

    for attempt in range(20):  # 增加尝试次数
        op = random.choice(operators)

        # 构建左右子树
        left_expr = build_expression(bound, max_depth - 1)
        right_expr = build_expression(bound, max_depth - 1)

        # 对于减法和除法，可能需要调整操作数顺序
        if op == '-':
            left_val = left_expr.compute()
            right_val = right_expr.compute()
            if left_val < right_val:
                # 交换操作数
                left_expr, right_expr = right_expr, left_expr

        try:
            candidate = CalcNode(lhs=left_expr, oper=op, rhs=right_expr)

            # 验证运算符数量
            if candidate.op_count() > 3:
                continue

            # 验证减法约束
            if not candidate.validate_subtraction():
                continue

            # 验证除法约束
            if not candidate.validate_division():
                continue

            # 验证最终结果非负
            result = candidate.compute()
            if result < 0:
                continue

            return candidate
        except:
            continue

    # 如果多次尝试失败，返回简单表达式
    return CalcNode(const=make_random_value(bound))


# -----------------------------
# 生成不重复题目
# -----------------------------
def produce_problems(count: int, range_limit: int) -> List[CalcNode]:
    problem_list = []
    seen_signatures: Set[str] = set()
    tries = 0
    max_tries = count * 50

    while len(problem_list) < count and tries < max_tries:
        tries += 1

        # 强制生成包含运算符的表达式
        expr = build_expression(range_limit, 2, force_operator=True)

        # 验证基本要求
        if expr.op_count() == 0 or expr.op_count() > 3:
            continue

        try:
            value = expr.compute()
            if value < 0:
                continue

            # 验证所有约束
            if not expr.validate_subtraction():
                continue
            if not expr.validate_division():
                continue

        except:
            continue

        # 检查重复
        sig = expr.unique_key()
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            problem_list.append(expr)

    # 如果生成数量不足，补充简单题目
    while len(problem_list) < count:
        simple_expr = build_simple_expression(range_limit)
        sig = simple_expr.unique_key()
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            problem_list.append(simple_expr)

    return problem_list[:count]


def build_simple_expression(bound: int) -> CalcNode:
    """构建简单的两操作数表达式"""
    operators = ['+', '*', '-']  # 简单表达式避免除法

    for attempt in range(10):
        op = random.choice(operators)
        left = CalcNode(const=make_random_value(bound))
        right = CalcNode(const=make_random_value(bound))

        if op == '-':
            left_val = left.compute()
            right_val = right.compute()
            if left_val < right_val:
                left, right = right, left

        try:
            candidate = CalcNode(lhs=left, oper=op, rhs=right)
            result = candidate.compute()
            if result >= 0:
                return candidate
        except:
            continue

    # 默认返回加法
    return CalcNode(
        lhs=CalcNode(const=make_random_value(bound)),
        oper='+',
        rhs=CalcNode(const=make_random_value(bound))
    )


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
        total_numer = w * int(denom) + int(numer)
        return Fraction(sign * total_numer, int(denom))
    elif '/' in s:
        n, d = s.split('/')
        return Fraction(int(n), int(d))
    else:
        return Fraction(int(s))


def safe_eval_expression(expr: str) -> Fraction:
    """安全计算表达式，完全使用分数运算"""
    # 替换运算符
    expr = expr.replace('×', '*').replace('÷', '/')

    # 处理带分数
    expr = re.sub(r"(-?\d+)'(\d+)/(\d+)", r"Fraction(\1 * \3 + \2, \3)", expr)

    # 处理普通分数 - 使用更精确的匹配
    # 先处理带分数，然后处理普通分数
    expr = re.sub(r'\b(\d+)/(\d+)\b', r"Fraction(\1, \2)", expr)

    # 确保所有数字都被转换为Fraction
    expr = re.sub(r'\b(\d+)\b', r"Fraction(\1)", expr)

    # 添加必要的括号
    tokens = tokenize_expression(expr)
    expr_with_parens = add_parentheses_by_priority(tokens)

    try:
        result = eval(expr_with_parens, {"Fraction": Fraction})
        # 确保结果是Fraction类型
        if isinstance(result, (int, float)):
            return Fraction(result).limit_denominator()
        return result
    except Exception as e:
        print(f"计算表达式错误: {expr}, 错误: {e}")
        # 如果eval失败，使用手动计算
        return manual_calculate(expr)


def tokenize_expression(expr: str) -> List[str]:
    """将表达式分解为token"""
    tokens = []
    current = ""

    for char in expr:
        if char in '()+-*/ ':
            if current:
                tokens.append(current)
                current = ""
            if char != ' ':
                tokens.append(char)
        else:
            current += char

    if current:
        tokens.append(current)

    return tokens


def add_parentheses_by_priority(tokens: List[str]) -> str:
    """根据运算符优先级添加括号"""
    # 简化的处理：为乘除运算添加括号
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ['*', '/'] and i > 0 and i < len(tokens) - 1:
            # 找到乘除运算的操作数
            left = find_operand(tokens, i - 1, -1)
            right = find_operand(tokens, i + 1, 1)

            # 添加括号
            if left[0] > 0 and tokens[left[0] - 1] != '(':
                result[left[0]] = '(' + result[left[0]]
                result[right[1]] = result[right[1]] + ')'

        result.append(token)
        i += 1

    return ' '.join(result)


def find_operand(tokens: List[str], start: int, direction: int) -> Tuple[int, int]:
    """找到操作数的范围"""
    if direction > 0:
        end = start
        while end < len(tokens) and tokens[end] not in ['+', '-', '*', '/', ')']:
            end += 1
        return start, end - 1
    else:
        end = start
        while end >= 0 and tokens[end] not in ['+', '-', '*', '/', '(']:
            end -= 1
        return end + 1, start


def manual_calculate(expr: str) -> Fraction:
    """手动计算表达式（处理eval失败的情况）"""
    # 移除所有空格
    expr = expr.replace(' ', '')

    # 处理括号
    while '(' in expr:
        start = expr.rfind('(')
        end = expr.find(')', start)
        if end == -1:
            break

        inner = expr[start + 1:end]
        inner_result = calculate_simple_expression(inner)
        expr = expr[:start] + str(inner_result.numerator) + '/' + str(inner_result.denominator) + expr[end + 1:]

    return calculate_simple_expression(expr)


def calculate_simple_expression(expr: str) -> Fraction:
    """计算简单表达式（无括号）"""
    # 处理乘除
    tokens = re.split(r'([*/])', expr)
    if len(tokens) > 1:
        result = parse_fraction(tokens[0])
        i = 1
        while i < len(tokens):
            op = tokens[i]
            next_val = parse_fraction(tokens[i + 1])
            if op == '*':
                result *= next_val
            else:  # '/'
                if next_val == 0:
                    raise ZeroDivisionError("除数为零")
                result /= next_val
            i += 2
        return result

    # 处理加减
    tokens = re.split(r'([+-])', expr)
    if len(tokens) > 1:
        result = parse_fraction(tokens[0])
        i = 1
        while i < len(tokens):
            op = tokens[i]
            next_val = parse_fraction(tokens[i + 1])
            if op == '+':
                result += next_val
            else:  # '-'
                result -= next_val
            i += 2
        return result

    return parse_fraction(expr)


def parse_fraction(expr: str) -> Fraction:
    """解析分数或整数"""
    expr = expr.strip()
    if '/' in expr:
        parts = expr.split('/')
        if len(parts) == 2:
            return Fraction(int(parts[0]), int(parts[1]))
    return Fraction(int(expr))

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
            ex_clean = ex_line.strip()
            if ex_clean.endswith('='):
                ex_clean = ex_clean[:-1].strip()
            if '. ' in ex_clean:
                expr_raw = ex_clean.split('. ', 1)[1]
            else:
                expr_raw = ex_clean

            ans_clean = ans_line.strip()
            if '. ' in ans_clean:
                ans_part = ans_clean.split('. ', 1)[1]
            else:
                ans_part = ans_clean

            try:
                # 使用安全的表达式计算
                computed = safe_eval_expression(expr_raw)
                user_val = decode_answer(ans_part)
                if computed == user_val:
                    correct_ids.append(idx)
                else:
                    print(f"题目 {idx} 计算不一致: 程序={computed}, 答案={user_val}, 表达式={expr_raw}")
                    wrong_ids.append(idx)
            except Exception as e:
                print(f"题目 {idx} 计算错误: {e}, 表达式={expr_raw}")
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
            exf.write(f"{idx}. {expr_text} =\n")
            ans_val = item.compute()
            anf.write(f"{idx}. {render_frac(ans_val)}\n")


if __name__ == '__main__':
    run_app()