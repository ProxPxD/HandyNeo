from typing import Iterable

from parameterized import parameterized
from py2neo import Node

from src.handyneo import N, R, RR
from tests.abstractTest import AbstractTest


class RelationTest(AbstractTest):

    nodes = {node.name: node for node in (N('Ania'), N('Phil'), N('Kate'), N('Yuan'))}

    @parameterized.expand([
        ('N_N', 'knows', [(nodes['Ania'], nodes['Phil'])]),
        ('Node_Node', 'knows', [(nodes['Ania'].node, nodes['Phil'].node)]),
        ('Node_N', 'knows', [(nodes['Ania'].node, nodes['Phil'])]),
        ('many_different', 'knowns', [(nodes['Ania'], nodes['Phil']), (nodes['Phil'].node, nodes['Kate']), (nodes['Kate'].node, nodes['Yuan'].node)]),
    ])
    def test_relation_creation(self, test_name: str, r_name: str, node_pairs: Iterable[tuple[N | Node, N | Node]]):
        r = R(r_name)
        self.assertIn(r_name, r.name)
        self.assertIn(r_name, RR.dir())
        self.assertIn(r, RR.vals())

        rels = [r(n1, n2) for n1, n2 in node_pairs]
        self.assertListEqual(rels, RR.get(r.name).children)
