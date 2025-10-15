"""
小学四则运算题目生成器
该模块用于生成符合小学教学要求的四则运算题目，支持自然数、真分数和带分数运算，
确保题目不重复且符合数学教学规范。
"""
import argparse
import random
import re
from fractions import Fraction
from typing import List, Set, Tuple


# 预编译正则表达式以提高性能
MIXED_FRACTION_PATTERN = re.compile(r"(-?\d+)'(\d+)/(\d+)")
FRACTION_PATTERN = re.compile(r'\b(\d+)/(\d+)\b')
INTEGER_PATTERN = re.compile(r'\b(\d+)\b')
OPERATOR_SPLIT_PATTERN = re.compile(r'([*/])')
PLUS_MINUS_SPLIT_PATTERN = re.compile(r'([+-])')


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
        if self.oper == '+':
            return left_val + right_val
        if self.oper == '-':
            return left_val - right_val
        if self.oper == '*':
            return left_val * right_val
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

        need_left_paren = self._need_parentheses(self.lhs, True)
        need_right_paren = self._need_parentheses(self.rhs, False)

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
            a, b = sorted([left_key, right_key])
            return f"{a}{self.oper}{b}"
        return f"{left_key}{self.oper}{right_key}"

    def op_count(self) -> int:
        """计算运算符数量"""
        if self.const is not None:
            return 0
        return 1 + self.lhs.op_count() + self.rhs.op_count()

    def validate_subtraction(self) -> bool:
        """验证所有减法操作 e1 - e2 满足 e1 ≥ e2"""
        if self.const is not None:
            return True

        if self.oper == '-':
            left_val = self.lhs.compute()
            right_val = self.rhs.compute()
            if left_val < right_val:
                return False

        return (self.lhs.validate_subtraction() and
                self.rhs.validate_subtraction())

    def validate_division(self) -> bool:
        """验证所有除法操作 e1 ÷ e2 的结果是真分数"""
        if self.const is not None:
            return True

        if self.oper == '/':
            left_val = self.lhs.compute()
            right_val = self.rhs.compute()
            if right_val == 0:
                return False
            result = left_val / right_val
            if result <= 0 or result >= 1:
                return False

        return (self.lhs.validate_division() and
                self.rhs.validate_division())


def make_random_value(bound: int) -> Fraction:
    """生成随机数值"""
    if bound <= 1:
        return Fraction(0)
    if random.random() < 0.5:
        return Fraction(random.randint(0, bound - 1))
    denom = random.randint(2, bound - 1)
    numer = random.randint(1, denom - 1)
    return Fraction(numer, denom)


def build_expression(bound: int, max_depth: int, force_operator=False) -> CalcNode:
    """递归构建算术表达式"""
    if max_depth <= 0 or (not force_operator and random.random() < 0.3):
        return CalcNode(const=make_random_value(bound))

    operators = ['+', '-', '*', '/']

    for _ in range(20):
        op = random.choice(operators)
        left_expr = build_expression(bound, max_depth - 1)
        right_expr = build_expression(bound, max_depth - 1)

        if op == '-':
            left_val = left_expr.compute()
            right_val = right_expr.compute()
            if left_val < right_val:
                left_expr, right_expr = right_expr, left_expr

        try:
            candidate = CalcNode(lhs=left_expr, oper=op, rhs=right_expr)
            if candidate.op_count() > 3:
                continue
            if not candidate.validate_subtraction():
                continue
            if not candidate.validate_division():
                continue
            result = candidate.compute()
            if result < 0:
                continue
            return candidate
        except (ZeroDivisionError, ValueError):
            continue

    return CalcNode(const=make_random_value(bound))


def produce_problems(count: int, range_limit: int) -> List[CalcNode]:
    """生成指定数量的不重复题目"""
    problem_list = []
    seen_signatures = set()
    tries = 0
    max_tries = count * 50

    while len(problem_list) < count and tries < max_tries:
        tries += 1
        expr = build_expression(range_limit, 2, True)

        if expr.op_count() == 0 or expr.op_count() > 3:
            continue

        try:
            value = expr.compute()
            if value < 0:
                continue
            if not expr.validate_subtraction():
                continue
            if not expr.validate_division():
                continue
        except (ZeroDivisionError, ValueError):
            continue

        sig = expr.unique_key()
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            problem_list.append(expr)

    while len(problem_list) < count:
        simple_expr = build_simple_expression(range_limit)
        sig = simple_expr.unique_key()
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            problem_list.append(simple_expr)

    return problem_list[:count]


def build_simple_expression(bound: int) -> CalcNode:
    """构建简单的两操作数表达式"""
    operators = ['+', '*', '-']
    for _ in range(10):
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
        except (ZeroDivisionError, ValueError):
            continue
    return CalcNode(
        lhs=CalcNode(const=make_random_value(bound)),
        oper='+',
        rhs=CalcNode(const=make_random_value(bound))
    )


def decode_answer(text: str) -> Fraction:
    """解析用户答案字符串为Fraction对象"""
    s = text.strip()

    if not s:
        return Fraction(0)

    try:
        if "'" in s:
            parts = s.split("'")
            if len(parts) != 2:
                return Fraction(0)

            whole = parts[0]
            frac_part = parts[1]

            # 检查分数部分格式 - 必须包含 '/'
            if '/' not in frac_part:
                # 抛出明确的错误信息
                raise ValueError(f"带分数缺少分数部分，只有整数: {s}")

            frac_parts = frac_part.split('/')
            if len(frac_parts) != 2:
                return Fraction(0)

            numer, denom = frac_parts

            # 验证分母不为0
            if int(denom) == 0:
                return Fraction(0)

            sign = -1 if whole and whole.startswith('-') else 1
            w = abs(int(whole)) if whole and whole not in ('', '-') else 0
            total_numer = w * int(denom) + int(numer)
            return Fraction(sign * total_numer, int(denom))

        elif '/' in s:
            parts = s.split('/')
            if len(parts) != 2:
                return Fraction(0)
            n, d = parts
            if int(d) == 0:
                return Fraction(0)
            return Fraction(int(n), int(d))

        else:
            # 纯整数
            return Fraction(int(s))

    except ValueError as e:
        # 重新抛出这个错误，让主程序捕获
        raise ValueError(str(e))
    except Exception:
        return Fraction(0)


def safe_eval_expression(expr: str) -> Fraction:
    # 在函数内部导入，避免启动时加载
    from fractions import Fraction
    """安全计算表达式，完全使用分数运算"""
    if not expr or expr.isspace():
        raise ValueError("空表达式")

    try:
        expr = expr.replace('×', '*').replace('÷', '/')
        expr = MIXED_FRACTION_PATTERN.sub(r"Fraction(\1 * \3 + \2, \3)", expr)
        expr = FRACTION_PATTERN.sub(r"Fraction(\1, \2)", expr)
        expr = INTEGER_PATTERN.sub(r"Fraction(\1)", expr)

        tokens = tokenize_expression(expr)
        expr_with_parens = add_parentheses_by_priority(tokens)

        result = eval(expr_with_parens, {"Fraction": Fraction})
        if isinstance(result, (int, float)):
            return Fraction(result).limit_denominator()
        return result

    except ZeroDivisionError:
        raise ValueError("除零错误")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"表达式语法错误: {e}")
    except Exception as e:
        raise ValueError(f"计算错误: {e}")


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
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (token in ['*', '/'] and i > 0 and
            i < len(tokens) - 1):
            left = find_operand(tokens, i - 1, -1)
            right = find_operand(tokens, i + 1, 1)
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
        while (end < len(tokens) and
               tokens[end] not in ['+', '-', '*', '/', ')']):
            end += 1
        return start, end - 1
    end = start
    while (end >= 0 and
           tokens[end] not in ['+', '-', '*', '/', '(']):
        end -= 1
    return end + 1, start


def manual_calculate(expr: str) -> Fraction:
    """手动计算表达式"""
    expr = expr.replace(' ', '')
    while '(' in expr:
        start = expr.rfind('(')
        end = expr.find(')', start)
        if end == -1:
            break
        inner = expr[start + 1:end]
        inner_result = calculate_simple_expression(inner)
        expr = (expr[:start] + str(inner_result.numerator) + '/' +
                str(inner_result.denominator) + expr[end + 1:])
    return calculate_simple_expression(expr)


def calculate_simple_expression(expr: str) -> Fraction:
    """计算简单表达式"""
    tokens = OPERATOR_SPLIT_PATTERN.split(expr)
    if len(tokens) > 1:
        result = parse_fraction(tokens[0])
        i = 1
        while i < len(tokens):
            op = tokens[i]
            next_val = parse_fraction(tokens[i + 1])
            if op == '*':
                result *= next_val
            else:
                if next_val == 0:
                    raise ZeroDivisionError("除数为零")
                result /= next_val
            i += 2
        return result

    tokens = PLUS_MINUS_SPLIT_PATTERN.split(expr)
    if len(tokens) > 1:
        result = parse_fraction(tokens[0])
        i = 1
        while i < len(tokens):
            op = tokens[i]
            next_val = parse_fraction(tokens[i + 1])
            if op == '+':
                result += next_val
            else:
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


def write_problems_batch(quiz_set: List[CalcNode]):
    """批量写入题目和答案"""
    exercise_lines = []
    answer_lines = []

    # 预先构建所有行内容
    for idx, item in enumerate(quiz_set, start=1):
        expr_text = item.to_text()
        ans_val = item.compute()
        exercise_lines.append(f"{idx}. {expr_text} =\n")
        answer_lines.append(f"{idx}. {render_frac(ans_val)}\n")

    # 一次性写入文件
    with open('Exercises.txt', 'w', encoding='utf-8') as exf:
        exf.writelines(exercise_lines)
    with open('Answers.txt', 'w', encoding='utf-8') as anf:
        anf.writelines(answer_lines)


def check_answers_batch(exercises: List[str], answers: List[str]):
    """批量检查答案"""
    correct_ids = []
    wrong_ids = []

    # 预处理所有题目和答案
    processed_data = []
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

        processed_data.append((idx, expr_raw, ans_part))

    # 批量计算
    results = []
    for idx, expr_raw, ans_part in processed_data:
        try:
            computed = safe_eval_expression(expr_raw)
            user_val = decode_answer(ans_part)
            results.append((idx, computed, user_val, None))
        except Exception as e:
            results.append((idx, None, None, e))

    # 批量比较结果
    for idx, computed, user_val, error in results:
        if error:
            print(f"题目 {idx} 错误: {error}")
            wrong_ids.append(idx)
        elif computed == user_val:
            correct_ids.append(idx)
        else:
            print(f"题目 {idx} 计算不一致: 表达式 = {expr_raw}")
            wrong_ids.append(idx)

    return correct_ids, wrong_ids

def read_files_batch(exercise_file: str, answer_file: str):
    """批量读取文件"""
    with open(exercise_file, 'r', encoding='utf-8') as ef:
        exercises = [line.strip() for line in ef if line.strip()]
    with open(answer_file, 'r', encoding='utf-8') as af:
        answers = [line.strip() for line in af if line.strip()]
    return exercises, answers

def write_grade_batch(correct_ids: List[int], wrong_ids: List[int]):
    """批量写入评分结果"""
    with open('Grade.txt', 'w', encoding='utf-8') as gf:
        gf.write(f"Correct: {len(correct_ids)} ({', '.join(map(str, correct_ids))})\n")
        gf.write(f"Wrong: {len(wrong_ids)} ({', '.join(map(str, wrong_ids))})\n")

def run_app():
    """主应用程序入口"""
    parser = argparse.ArgumentParser(prog="小学算术题生成器")
    parser.add_argument('-n', type=int, help='题目数量')
    parser.add_argument('-r', type=int, help='数值上限')
    parser.add_argument('-e', type=str, help='题目文件')
    parser.add_argument('-a', type=str, help='答案文件')
    opts = parser.parse_args()

    if opts.e and opts.a:
        # 一次性读取所有内容
        exercises, answers = read_files_batch(opts.e, opts.a)

        # 批量处理
        correct_ids, wrong_ids = check_answers_batch(exercises, answers)

        # 一次性写入结果
        write_grade_batch(correct_ids, wrong_ids)
        return

    if opts.n is None or opts.r is None:
        print("错误：必须同时指定 -n 和 -r 参数")
        return

    if opts.r < 1:
        print("错误：-r 必须为正整数")
        return

    quiz_set = produce_problems(opts.n, opts.r)
    with open('Exercises.txt', 'w', encoding='utf-8') as exf, \
         open('Answers.txt', 'w', encoding='utf-8') as anf:
        for idx, item in enumerate(quiz_set, 1):
            expr_text = item.to_text()
            exf.write(f"{idx}. {expr_text} =\n")
            ans_val = item.compute()
            anf.write(f"{idx}. {render_frac(ans_val)}\n")


if __name__ == '__main__':
    run_app()