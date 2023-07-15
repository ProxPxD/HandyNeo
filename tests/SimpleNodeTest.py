from typing import Iterable

from parameterized import parameterized
from more_itertools import distinct_combinations, flatten
from py2neo import Node

from src.handyneo import Nabel, Nabels, N
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

    @classmethod
    def gen_all_simple_node_creation_test(cls):
        for pair in cls.mixed:
            name = 'test_simple_' + '_'.join(map(lambda e: type(e).__name__, pair))
            test = cls.gen_test_simple_node_creation(name, pair)
            setattr(SimpleNodeTest, name, test)

    @classmethod
    def gen_test_simple_node_creation(cls, name: str, nabels: Nabels):
        def test(self, *args):
            e_labels = set(col_to_str(nabels))
            try:
                n = N(name, *nabels)
                self.assertEqual(name, n.name)
                self.assertIsInstance(n.node, Node)
                self.assertSetEqual(e_labels, n.labels)
            except Exception:
                self.fail('Unexpected exception')
        return test
