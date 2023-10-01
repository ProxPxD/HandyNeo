from typing import Optional

from parameterized import parameterized

from src.handyneo import Relationer, NodeMaker, NabelSome, R, N, S
from src.utils import save_iterabilize, save_flatten
from tests.abstractTest import AbstractTest


class NodeMakerTest(AbstractTest):

    @parameterized.expand([
        ('only_labels', ('Person', ), None, None),
        ('with_relationer', tuple(), Relationer(parents_rel=R('belongs-to'), parents_name_as_label=False), N('Jan', 'Polish', 'European')),
    ])
    def test_node_maker(self, name: str, labels: tuple[str, ...], relationer: Optional[Relationer], parents: Optional[NabelSome]):
        node_maker = NodeMaker(*labels, relationer=relationer)
        save_parents = save_iterabilize(parents)
        parents_labels = save_flatten(save_parents, list, map_func=lambda p: p.labels)

        child = node_maker(name, parents=parents)

        if not relationer:
            self.assertCountEqual(labels, child.labels)
        if relationer:
            self.assertEqual(relationer, node_maker.rel)
            self.assertCountEqual(list(labels) + parents_labels, child.labels)
            relationships = relationer.get_rel(S.parents).children
            start_nodes = set(map(lambda r: r.start_node, relationships))
            end_nodes = set(map(lambda r: r.end_node, relationships))
            self.assertIn(child.node, end_nodes)
            for parent in save_parents:
                self.assertIn(parent.node, start_nodes)


