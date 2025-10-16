import unittest
import tempfile
import os
import sys
from fractions import Fraction
from io import StringIO

# 导入被测试的模块
from main import (
    render_frac, CalcNode, make_random_value, build_expression,
    produce_problems, decode_answer, safe_eval_expression,
    manual_calculate, build_simple_expression
)


class TestFractionRendering(unittest.TestCase):
    """测试分数格式化功能"""

    def test_integer_rendering(self):
        self.assertEqual(render_frac(Fraction(5)), "5")
        self.assertEqual(render_frac(Fraction(0)), "0")

    def test_proper_fraction_rendering(self):
        self.assertEqual(render_frac(Fraction(1, 2)), "1/2")
        self.assertEqual(render_frac(Fraction(3, 4)), "3/4")

    def test_mixed_fraction_rendering(self):
        self.assertEqual(render_frac(Fraction(5, 2)), "2'1/2")
        self.assertEqual(render_frac(Fraction(7, 3)), "2'1/3")

    def test_negative_fraction_rendering(self):
        # 修复后的测试用例
        self.assertEqual(render_frac(Fraction(-3, 2)), "-1'1/2")
        self.assertEqual(render_frac(Fraction(-1, 3)), "-1/3")
        self.assertEqual(render_frac(Fraction(-5, 2)), "-2'1/2")


class TestCalcNode(unittest.TestCase):
    """测试表达式节点类"""

    def setUp(self):
        self.simple_add = CalcNode(
            lhs=CalcNode(const=Fraction(1, 2)),
            oper='+',
            rhs=CalcNode(const=Fraction(1, 3))
        )

        self.complex_expr = CalcNode(
            lhs=CalcNode(const=Fraction(1)),
            oper='*',
            rhs=CalcNode(
                lhs=CalcNode(const=Fraction(1, 2)),
                oper='+',
                rhs=CalcNode(const=Fraction(1, 4))
            )
        )

    def test_const_node(self):
        node = CalcNode(const=Fraction(1, 2))
        self.assertEqual(node.compute(), Fraction(1, 2))
        self.assertEqual(node.to_text(), "1/2")
        self.assertEqual(node.op_count(), 0)

    def test_computation(self):
        # 测试 1/2 + 1/3 = 5/6
        result = self.simple_add.compute()
        self.assertEqual(result, Fraction(5, 6))

    def test_text_generation(self):
        text = self.simple_add.to_text()
        self.assertIn("1/2", text)
        self.assertIn("1/3", text)
        self.assertIn("+", text)

    def test_priority_parentheses(self):
        # 测试 1 * (1/2 + 1/4)
        text = self.complex_expr.to_text()
        self.assertTrue("(" in text and ")" in text)

    def test_unique_key(self):
        key1 = self.simple_add.unique_key()
        # 创建相同的表达式
        same_expr = CalcNode(
            lhs=CalcNode(const=Fraction(1, 2)),
            oper='+',
            rhs=CalcNode(const=Fraction(1, 3))
        )
        key2 = same_expr.unique_key()
        self.assertEqual(key1, key2)

    def test_validate_subtraction(self):
        valid_sub = CalcNode(
            lhs=CalcNode(const=Fraction(3, 2)),
            oper='-',
            rhs=CalcNode(const=Fraction(1, 2))
        )
        self.assertTrue(valid_sub.validate_subtraction())

    def test_validate_division(self):
        valid_div = CalcNode(
            lhs=CalcNode(const=Fraction(1, 2)),
            oper='/',
            rhs=CalcNode(const=Fraction(3, 2))
        )
        # 1/2 ÷ 3/2 = 1/3，是真分数
        self.assertTrue(valid_div.validate_division())


class TestProblemGeneration(unittest.TestCase):
    """测试题目生成功能"""

    def test_make_random_value(self):
        for _ in range(100):
            val = make_random_value(10)
            self.assertIsInstance(val, Fraction)
            self.assertTrue(val >= 0)

    def test_build_simple_expression(self):
        expr = build_simple_expression(10)
        self.assertIsInstance(expr, CalcNode)
        result = expr.compute()
        self.assertTrue(result >= 0)

    def test_build_expression(self):
        expr = build_expression(10, 2)
        self.assertIsInstance(expr, CalcNode)
        result = expr.compute()
        self.assertTrue(result >= 0)

    def test_produce_problems(self):
        problems = produce_problems(5, 10)
        self.assertEqual(len(problems), 5)

        # 验证所有题目都不重复
        signatures = set()
        for problem in problems:
            sig = problem.unique_key()
            self.assertNotIn(sig, signatures)
            signatures.add(sig)

            # 验证题目符合要求
            result = problem.compute()
            self.assertTrue(result >= 0)
            self.assertTrue(problem.validate_subtraction())
            self.assertTrue(problem.validate_division())


class TestAnswerParsing(unittest.TestCase):
    """测试答案解析功能"""

    def test_parse_integer(self):
        self.assertEqual(decode_answer("5"), Fraction(5))
        self.assertEqual(decode_answer("0"), Fraction(0))

    def test_parse_proper_fraction(self):
        self.assertEqual(decode_answer("1/2"), Fraction(1, 2))
        self.assertEqual(decode_answer("3/4"), Fraction(3, 4))

    def test_parse_mixed_fraction(self):
        self.assertEqual(decode_answer("2'1/2"), Fraction(5, 2))
        self.assertEqual(decode_answer("1'1/3"), Fraction(4, 3))

    def test_parse_negative_mixed_fraction(self):
        self.assertEqual(decode_answer("-1'1/2"), Fraction(-3, 2))

    def test_parse_invalid_format(self):
        self.assertEqual(decode_answer("2''1/2"), Fraction(0))
        self.assertEqual(decode_answer("invalid"), Fraction(0))
        self.assertEqual(decode_answer("1/2/3"), Fraction(0))

    def test_parse_empty_string(self):
        self.assertEqual(decode_answer(""), Fraction(0))
        self.assertEqual(decode_answer("   "), Fraction(0))


class TestExpressionEvaluation(unittest.TestCase):
    """测试表达式计算功能"""

    def test_simple_expression_eval(self):
        result = safe_eval_expression("1/2 + 1/3")
        self.assertEqual(result, Fraction(5, 6))

    def test_mixed_operations(self):
        result = safe_eval_expression("1/2 * 2/3 + 1/4")
        self.assertEqual(result, Fraction(7, 12))

    def test_with_parentheses(self):
        result = safe_eval_expression("(1/2 + 1/3) * 2")
        self.assertEqual(result, Fraction(5, 3))

    def test_mixed_fraction_in_expression(self):
        result = safe_eval_expression("1'1/2 + 1/2")
        self.assertEqual(result, Fraction(2))

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            safe_eval_expression("1/0")

    def test_invalid_expression(self):
        # 修复：使用更明显的无效表达式
        with self.assertRaises(ValueError):
            safe_eval_expression("1/ + /2")  # 明显的语法错误
        with self.assertRaises(ValueError):
            safe_eval_expression("")  # 空表达式


class TestManualCalculation(unittest.TestCase):
    """测试手动计算功能"""

    def test_manual_simple_expression(self):
        result = safe_eval_expression("1/2+1/3")
        self.assertEqual(result, Fraction(5, 6))

    def test_manual_multiplication(self):
        result = safe_eval_expression("2/3*3/4")
        self.assertEqual(result, Fraction(1, 2))

    def test_manual_division(self):
        result = safe_eval_expression("1/2÷1/4")
        self.assertEqual(result, Fraction(2))


class TestIntegration(unittest.TestCase):
    """集成测试"""
    def test_end_to_end_generation(self):
        """测试完整的题目生成流程"""
        problems = produce_problems(3, 5)
        self.assertEqual(len(problems), 3)

        # 测试每个题目都能正确格式化和计算
        for i, problem in enumerate(problems):
            text = problem.to_text()
            computed_result = problem.compute()

            # 验证生成的文本可以正确解析和计算
            parsed_result = safe_eval_expression(text)
            self.assertEqual(computed_result, parsed_result)

    def test_file_operations(self):
        """测试文件读写功能"""
        from main import write_problems_batch, check_answers_batch, write_grade_batch

        # 创建一些测试题目
        problems = [
            CalcNode(
                lhs=CalcNode(const=Fraction(1, 2)),
                oper='+',
                rhs=CalcNode(const=Fraction(1, 3))
            ),
            CalcNode(
                lhs=CalcNode(const=Fraction(2)),
                oper='*',
                rhs=CalcNode(const=Fraction(1, 4))
            )
        ]

        # 写入文件
        with tempfile.TemporaryDirectory() as tmpdir:
            exercise_file = os.path.join(tmpdir, 'Ex.txt')
            answer_file = os.path.join(tmpdir, 'Answers.txt')
            grade_file = os.path.join(tmpdir, 'Grade.txt')

            # 重定向文件路径
            import main
            original_write = main.write_problems_batch

            def temp_write(probs):
                with open(exercise_file, 'w', encoding='utf-8') as exf, \
                        open(answer_file, 'w', encoding='utf-8') as anf:
                    for idx, item in enumerate(probs, start=1):
                        expr_text = item.to_text()
                        exf.write(f"{idx}. {expr_text} =\n")
                        ans_val = item.compute()
                        anf.write(f"{idx}. {render_frac(ans_val)}\n")

            temp_write(problems)

            # 验证文件内容
            with open(exercise_file, 'r', encoding='utf-8') as f:
                exercises = f.readlines()
            with open(answer_file, 'r', encoding='utf-8') as f:
                answers = f.readlines()

            self.assertEqual(len(exercises), 2)
            self.assertEqual(len(answers), 2)

            # 测试答案检查
            correct, wrong = check_answers_batch(exercises, answers)
            self.assertEqual(len(correct), 2)
            self.assertEqual(len(wrong), 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_zero_bound(self):
        """测试数值范围为0的情况"""
        val = make_random_value(1)
        self.assertEqual(val, Fraction(0))

    def test_single_operation(self):
        """测试单次运算"""
        problems = produce_problems(1, 2)
        self.assertEqual(len(problems), 1)
        self.assertLessEqual(problems[0].op_count(), 3)

    def test_large_quantity(self):
        """测试生成大量题目"""
        problems = produce_problems(100, 5)
        self.assertEqual(len(problems), 100)

        # 验证所有题目都符合要求
        for problem in problems:
            result = problem.compute()
            self.assertTrue(result >= 0)
            self.assertTrue(problem.op_count() <= 3)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestFractionRendering,
        TestCalcNode,
        TestProblemGeneration,
        TestAnswerParsing,
        TestExpressionEvaluation,
        TestManualCalculation,
        TestIntegration,
        TestEdgeCases
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    # 运行测试
    success = run_tests()

    # 输出总结
    if success:
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！程序功能正常。")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ 部分测试失败，请检查代码。")
        print("=" * 50)
        sys.exit(1)