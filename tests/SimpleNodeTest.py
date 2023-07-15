from itertools import product

from more_itertools import distinct_combinations, flatten
from parameterized import parameterized
from py2neo import Node

from src.handyneo import Nabels, N
from src.utils import col_to_str
from tests.abstractTest import AbstractTest


class SimpleNodeTest(AbstractTest):

    is_generated = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_generated:
            self.gen_all_simple_node_creation_test()
            self.is_generated = True

    strs = ('A', 'B')
    ns = (N('A'), N('B'))
    nodes = (Node(name='A'), Node(name='B'))
    mixed = list(filter(lambda pair: col_to_str(pair[0]) != col_to_str(pair[1]), distinct_combinations(flatten((strs, ns, nodes)), 2)))

    node_kwargs = ({'val': 44}, {})

    @classmethod
    def gen_all_simple_node_creation_test(cls):
        for pair, kwargs in product(cls.mixed, cls.node_kwargs):
            name = 'test_simple_' + '_'.join(map(lambda e: type(e).__name__, pair)) + ('' if not kwargs else '_with_kwargs')
            test = cls.gen_test_simple_node_creation(name, pair, kwargs)
            setattr(SimpleNodeTest, name, test)

    @classmethod
    def gen_test_simple_node_creation(cls, name: str, nabels: Nabels, kwargs):
        def test(self, *args):
            e_labels = set(col_to_str(nabels))
            try:
                n = N(name, *nabels, **kwargs)
                self.assertEqual(name, n.name)
                self.assertIsInstance(n.node, Node)
                self.assertSetEqual(e_labels, n.labels)
                for k, v in kwargs.items():
                    self.assertIn(k, n.node)
                    self.assertEqual(v, n.node[k])
                    self.assertIn(k, n.keys())
                    self.assertEqual(v, n[k])
            except Exception:
                self.fail('Unexpected exception')
        return test

    @parameterized.expand([
        ('str', 'A', 'A'),
        ('node', Node(name='A'), 'A'),
        ('N', N('A'), 'A'),
    ])
    def test_col_to_str(self, name, arg, expected):
        actual = col_to_str(arg)
        self.assertEqual(expected, actual)