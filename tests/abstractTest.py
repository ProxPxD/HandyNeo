import abc
import unittest
from dataclasses import dataclass
from typing import Iterable
from colorama import Fore, Back, Style


@dataclass
class Status:
    PASS = 'PASS'
    FAIL = 'FAIL'
    SKIP = 'SKIP'
    ERROR = 'ERROR'


class AbstractTest(unittest.TestCase, abc.ABC):
    half_sep_length = 40
    currentResult = None
    status_distance = int(0.75 * 2*half_sep_length)

    total = 0
    failure = 0
    errors = 0
    skipped = 0

    @classmethod
    def print_sep_with_text(cls, text: str, sep: str = '*') -> None:
        with_sep_lines = sep * cls.half_sep_length + f' {text} ' + sep * cls.half_sep_length
        over_length = len(with_sep_lines) - cls.half_sep_length*2
        to_print = with_sep_lines[over_length//2 : -over_length//2]
        print(to_print)

    @classmethod
    def setUpClass(cls) -> None:
        cls.print_sep_with_text(f'Starting {cls._get_test_name()} tests!')

    def setUp(self) -> None:
        if not self.get_method_name().startswith('test_'):
            return
        super().setUp()
        print('- ', self.get_method_name(), end=' ... ')

    def tearDown(self) -> None:
        if not self.get_method_name().startswith('test_'):
            return
        super().tearDown()
        result = self.get_test_result()

        is_error = any(test == self for test, text in result.errors)
        is_failure = any(test == self for test, text in result.failures)
        is_skipped = any(test == self for test, text in result.skipped)
        passed = not (is_error or is_failure or is_skipped)

        self.__class__.total += 1
        if is_error:
            self.__class__.errors += 1
        if is_failure:
            self.__class__.failure += 1
        if is_skipped:
            self.__class__.skipped += 1

        test_printing_length = len(self.get_method_name()) + 5
        padding = self.status_distance - test_printing_length
        status = Status.PASS if passed else Status.ERROR if is_error else Status.FAIL if is_failure else Status.SKIP if is_skipped else \
        'WRONG UNIT TEST OUTCOME CHECKING! Investigate (possible incompatible with a python newer than 3.10)'
        status = self.colorize(status)
        print(f'{" "*padding}{status}')

    def get_test_result(self):
        if hasattr(self._outcome, 'errors'):
            result = self._get_legacy_test_results()
        else:  # Python 3.11+
            result = self._outcome.result
        return result

    def _get_legacy_test_results(self):
        ''' Python 3.4 - 3.10  (These two methods have no side effects) '''
        result = self.defaultTestResult()
        self._feedErrorsToResult(result, self._outcome.errors)
        for test, reason in self._outcome.skipped:
            self._addSkip(result, test, reason)
        return result

    def colorize(self, to_color: str):
        match to_color:
            case Status.PASS:  color = Fore.LIGHTGREEN_EX
            case Status.FAIL:  color = Fore.MAGENTA
            case Status.SKIP:  color = Fore.LIGHTYELLOW_EX
            case Status.ERROR: color = Fore.RED
            case _: color = ''
        return f'{color}{to_color}{Style.RESET_ALL}'

    @classmethod
    def tearDownClass(cls) -> None:
        cls.print_statistics(percentage=False)

    @classmethod
    def print_statistics(cls, failure=None, errors=None, skipped=None, total=None, *, short=False, percentage=True):
        if failure is None:
            failure = cls.failure
        if errors is None:
            errors = cls.errors
        if skipped is None:
            skipped = cls.skipped
        if total is None:
            total = cls.total
        failed = failure + errors
        total_run = total - skipped
        passed = total_run - failed

        if total == 0:
            print('There are no tests')
            return

        if short:
            ef_division = f'{failure}F, {errors}E' if errors else f'{failed}F'
            statistics_str = f'({ef_division}, {passed}P)/{total_run}'
            if skipped:
                statistics_str += f',    {skipped}S'
        else:
            ef_division = f' (Failures: {failure}, Errors: {errors})' if errors else ''
            statistics_str = f'Failed: {failed}{ef_division}, Passed: {passed}, Total: {total_run}'
        if percentage:
            ef_division = f' (Failures: {100 * failure / total_run:.1f}%, Errors: {100 * errors / total_run:.1f}%)' if errors else ''
            statistics_str = f'Failed: {100 * failed / total_run:.1f}%{ef_division}, Passed: {100 * passed / total_run:.1f}%'

        if skipped and (not short or percentage):
            statistics_str += f'   (Skipped: {skipped})'

        print(statistics_str)

    def run(self, result: unittest.result.TestResult | None = ...) -> unittest.result.TestResult | None:
        self.currentResult = result
        unittest.TestCase.run(self, result)

    def get_method_name(self) -> str:
        return self.id().split('.')[-1]

    @classmethod
    def _get_test_name(cls) -> str:
        return cls.__name__.removesuffix('Test')

    def get_parameterized_methods_of_current_test(self, *method_nums) -> Iterable:
        if not method_nums:
            return iter([])

        method_name = self.get_method_name()
        method_prefix = f'{method_name}_'
        child_methods = [name for name in dir(self) if name.startswith(method_prefix)]

        if method_nums[0] is None:
            method_nums = range(len(child_methods))
        number_marks = (name.removeprefix(method_prefix).split('_')[0] for name in child_methods)
        zero_count = len(next((num for num in number_marks if all((ch == '0' for ch in num))), ''))

        method_prefixes = (f'{method_prefix}{str(i).zfill(zero_count)}' for i in method_nums)
        methods = (getattr(self, actual_name) for expected_prefix in method_prefixes for actual_name in child_methods
                   if actual_name.startswith(expected_prefix))
        return methods

    def run_current_test_with_params(self, *method_nums):  # TODO: support custom names
        '''
        Run in a method declared after the desired parametrized method and named the same as it
        :param method_nums:
        :return:
        '''
        for method in self.get_parameterized_methods_of_current_test(*method_nums):
            method()
