import operator as op
import random
from collections import ChainMap, namedtuple
from functools import reduce
from itertools import product, chain

from py2neo import Node, Relationship

from src.handyneo import Relationer, R, S, Filterer, RR, N, NN, to_name
from src.utils import save_iterabilize
from tests.abstractTest import AbstractTest

random.seed(49)

restriction_names = ['reversed', 'children', 'adding']
Restrictions = namedtuple('Restrictions', restriction_names, defaults=(False, ) * len(restriction_names))


class RelationerTest(AbstractTest):
    is_generated = False
    params = {}
    random_test_percentage = 0
    conf_percentage = .000_010  # 0.000_050
    arg_percentage = 1

    restrictions = Restrictions(
        reversed=False,
        children=True,
        adding=True,
        # TODO: labels
    )

    is_testing_only_reversed = False

    main_types = (S.parents, S.unnamed, S.kwargs)
    sec_types = (S.rel, S.name_as_label, S.labels_inherit, S.reversed)
    rel_vals = (lambda name: R(name, True), lambda name: Relationship.type(name))
    bool_vals = (True, False, None)

    classes = (lambda name, labels: N(name, *list(map(lambda l: l + f'---{name}', labels))), lambda name, labels: Node(*list(map(lambda l: l + f'---{name}', labels)), name=name))
    all_labels = (tuple(), ('A', 'B', 'C'))
    all_node_names = (tuple(), ('1', ), ('1', '2', '3'))
    node_types = (*main_types, S.children)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_generated:
            RelationerTest.params = {
                **{f'{main_type}_{sec_type}': [{f'{main_type}_{sec_type}': val} for val in (self.bool_vals if sec_type != S.rel else ([c(f'{main_type}_{sec_type}') for c in self.rel_vals] + [None]))]\
                   for main_type in self.main_types for sec_type in self.sec_types},
                **{node_type: [{node_type: tuple((c(node_type[0] + node_name + f'_{random.random()}', labels) for node_name in node_names)) if len(node_names) != 1 else c(node_type[0] + node_names[0] + f'_{random.random()}', labels)} for c in self.classes for node_names in self.all_node_names for labels in self.all_labels if not (labels and (not node_names or node_type == S.children))] \
                   for node_type in self.node_types},
            }

            RelationerTest.gen_all_relationer_tests()
            RelationerTest.is_generated = True

    @classmethod
    def gen_all_relationer_tests(cls):
        confs = {k:v for k, v in cls.params.items() if '_' in k}
        args = {k:v for k, v in cls.params.items() if '_' not in k}
        ProductParams = namedtuple('ProductParams', ['confs', 'args'])
        dictionize = lambda values, percentage=None: [dict(ChainMap(*pos_values_tuples).items()) for pos_values_tuples in product(*values) if percentage is None or random.random() < percentage]
        product_params = ProductParams(dictionize(confs.values(), cls.conf_percentage), dictionize(args.values(), cls.arg_percentage))
        print('Confgs: ', len(product_params.confs))
        print('Args: ', len(product_params.args))
        print('Pos: ', len(product_params.confs)*len(product_params.args))
        for conf_sample, arg_sample in product(product_params.confs, product_params.args):
            is_executing = True
            if cls.restrictions.children:
                is_executing &= bool(arg_sample[S.children])
            if cls.restrictions.adding:
                is_executing &= any((conf_sample[f'{key}_rel'] and arg_sample[key] for key in (S.kwargs, S.parents, S.unnamed)))
            if cls.restrictions.reversed:
                is_executing &= any((conf_sample[f'{key}_reversed'] and arg_sample[key] for key in (S.kwargs, S.parents, S.unnamed)))
            if not is_executing or random.random() < cls.random_test_percentage:
                continue
            test_name = 'test_with_' + reduce(op.add, (f'_and_{key}_as_{val if isinstance(val, bool) else len(val) if isinstance(val, tuple) else 1 if key in (S.parents, S.children, S.unnamed, S.kwargs) else val.__class__.__name__ }' if (val is not None and not (S.reversed in key and val is False)) else '' for key, val in chain(conf_sample.items(), arg_sample.items())), '')
            test_name += reduce(op.add, ((f'_and_{key}_labels' if arg_sample[key] and save_iterabilize(arg_sample[key])[0].labels else '') for key in cls.main_types), '')
            test_name = test_name.replace('_and_', '', 1).lower()
            test = cls.gen_relationer_test(test_name, conf_sample, arg_sample)
            setattr(RelationerTest, test_name, test)

    # TODO: add test for reverse et al.
    @classmethod
    def gen_relationer_test(cls, test_name, confs: dict[str, R | bool | None], args: dict):
        def test(self):
            relationer = Relationer(change_name=True, **confs)
            orig_args = args.copy()

            parents = args.get(S.parents)
            children = args.get(S.children)
            unnamed = args.get(S.unnamed)
            for key in (S.parents, S.children, S.unnamed):
                if key in args:
                    del args[key]
            to_list = lambda a: [a] if a else [None]
            positional_args = to_list(parents) + to_list(children) + to_list(unnamed)
            relationer(*positional_args, **args)
            rs = Filterer.filter_rel(confs, True)
            nals = Filterer.filter_name_as_labels(confs, True)
            lis = Filterer.filter_labels_inherit(confs, True)
            revs = Filterer.filter_reversed(confs, True)
            all_ns = args | {S.parents: parents, S.children: children, S.unnamed: unnamed}
            for name, r in rs.items():
                if not r:
                    continue
                r_type = name.removesuffix(f'_{S.rel}')
                r_confs = relationer.get_rel_confs(r_type)
                higher_nodes = save_iterabilize(all_ns[r_type])

                self.assertTrue(r_confs.rel.name.startswith(name))
                self.assertEqual(r_confs.rel, RR.get(r_confs.rel.name))
                self.assertNotEqual(r_confs.labels_inherit, None)
                if lis[f'{r_type}_{S.labels_inherit}'] is not None:
                    self.assertEqual(r_confs.labels_inherit, lis[f'{r_type}_{S.labels_inherit}'])
                self.assertNotEqual(r_confs.name_as_label, None)
                if nals[f'{r_type}_{S.name_as_label}'] is not None:
                    self.assertEqual(r_confs.name_as_label, nals[f'{r_type}_{S.name_as_label}'])
                self.assertNotEqual(r_confs.reversed, None)
                if revs[f'{r_type}_{S.reversed}'] is not None:
                    self.assertEqual(r_confs.reversed, revs[f'{r_type}_{S.reversed}'])
                if children is not None and r_type in all_ns:
                    children = save_iterabilize(children)
                    self.assertEqual(len(higher_nodes)*len(children), len(r_confs.rel.children))
                    start_col = list(map(to_name, higher_nodes))
                    end_col = list(map(to_name, children))
                    if r_confs.reversed:
                        start_col, end_col = end_col, start_col
                    for child_rel in r_confs.rel.children:
                        self.assertIn(child_rel.start_node[S.name], start_col)
                        self.assertIn(child_rel.end_node[S.name], end_col)

                    for child, higher_node in product(children, higher_nodes):
                        is_name_in = higher_node[S.name] in child.labels
                        self.assertTrue(is_name_in if r_confs.name_as_label else not is_name_in)
                        for higher_label in higher_node.labels:
                            is_label_in = higher_label in child.labels
                            self.assertTrue(is_label_in if r_confs.labels_inherit else not is_label_in)

            RR.clean_children()
            NN.clean_children()
        return test

    @classmethod
    def gen_test_relationer(self, test_name, confs: dict[str, R | bool | None], call_args: tuple, call_kwargs: dict):
        try:
            relationer = Relationer(**confs)
            relationer(*call_args, **call_kwargs)

            rs = Filterer.filter_rel(confs, True)
            parents = save_iterabilize(call_args[0])
            children = save_iterabilize(call_args[1])
            unnamed = call_args[2:]
            all_ns = call_kwargs | {S.parents: parents, S.unnamed: unnamed}
            for name, r in rs.items():
                r_type = name.removesuffix(f'_{S.rel}')
                self.assertIn(name, RR.dir().keys())
                self.assertEqual(r, RR.get(name))
                if r_type in all_ns:
                    self.assertEqual(len(all_ns[r_type])*len(children), len(r.children))
        except Exception:
            self.fail('Unexpected exception!')
