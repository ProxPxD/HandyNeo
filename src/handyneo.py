from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable, Type, Tuple

from py2neo import Graph, Relationship, Node

from src.Exceptions import MathematicBreakDownException
from src.utils import save_iterabilize, DictClass

Graph.create_all = lambda self, *subgraphs: [self.create(subgraph) for subgraph in subgraphs] is None and None
import operator as op


##################
# Implementation #
##################

@dataclass(frozen=True)
class Strings:
    reversed = 'reversed'
    name = 'name'
    parents = 'parents'
    children = 'children'
    unnamed = 'unnamed'
    rel = 'rel'
    name_as_label = 'name_as_label'
    labels_inherit = 'labels_inherit'
S = Strings


@dataclass
class D:
    _map_func: Callable

    @classmethod
    def dir(cls):
        return {name: val for name, val in cls.__dict__.items() if not name.startswith('_')}

    @classmethod
    def vals(cls):
        return cls.dir().values()

    @classmethod
    def pure_vals(cls, to: Callable = iter):
        return to((v for val in map(cls._map_func, cls.vals()) for v in (val if isinstance(val, list) else [val])))


class NN(D): pass
class RR(D): pass


class HasName:
    def __init__(self, name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name: str = name

    @abstractmethod
    @property
    def name(self) -> str:
        return self._name

    def get_name(self) -> str:
        return self.name


class R(HasName):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.rel: Type[Relationship] = Relationship.type(name)
        self.children: list[Relationship] = []
        setattr(RR, name, self)

    @property
    def name(self) -> str:
        return str(type(self.rel))

    def get_rel(self) -> Type[Relationship]:
        return self.rel

    def get_children(self) -> list[Relationship]:
        return self.children

    def __call__(self, *args, **kwargs) -> Relationship:
        r = self.rel(*args, **kwargs)
        self.children.append(r)
        return r

    def __repr__(self):
        return f'R-{self.name}'


class N(HasName):
    pass


N_potcol = Tuple[N] | N
Nabel = N | str  # node/label
Nabels = Iterable[Nabel]


@dataclass
class NodeVar:
    _values: Nabels | None = field(default=tuple)
    is_constant: bool = field(default=True)
    vals = property(
        lambda self: self._values,
        lambda self, val: setattr(self, '_values', save_iterabilize(val)) if not self.is_constant else None
    )

    def set_first_from(self, args: tuple) -> tuple:
        if not self.is_constant:
            self.vals = args[0]
            return args[1:]
        return args

    def __post_init__(self):
        self.vals = self._values


@dataclass
class NodeVars(DictClass):
    parents: NodeVar  = field(default=NodeVar(None, False))
    children: NodeVar = field(default=NodeVar(None, False))


@dataclass
class NabelConfig(DictClass):
    rel: R = field(default=None)
    name_as_label: bool = field(default=True)
    labels_inherit: bool = field(default=False)
    reversed: bool = field(default=False)

    @classmethod
    def make_from(cls, name: str, kwargs: dict, **defaults):
        params = {param_name.removeprefix(f'{name}_'): param_value for param_name, param_value in kwargs.items() if name in param_name}
        for default_key, default_value in defaults.items():
            if default_key not in params:
                params[default_key] = default_value
        return NabelConfig(**params)


class Relationer:
    '''
        [NAME]_rel, [NAME]_labels_inherit, [NAME]_name_as_label
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_vars = NodeVars()
        self._nabels_configs: dict[str, NabelConfig] = {
            S.parents: NabelConfig.make_from(S.unnamed, kwargs, labels_inherit=True),
            S.unnamed: NabelConfig.make_from(S.unnamed, kwargs),
        }
        truncated_keys = map(NabelConfig.map_to_contained_key, kwargs.keys())
        nonnone_keys = filter(bool, truncated_keys)
        unique_keys = set(nonnone_keys)
        for unique_key in unique_keys:
            self._nabels_configs[unique_key] = NabelConfig.make_from(unique_key, kwargs)

    def __call__(self, *labels, **named_nabels: Nabel):
        labels = self._node_vars.parents.set_first_from(labels)
        labels = self._node_vars.children.set_first_from(labels)
        self._rel(self._node_vars.parents.vals, S.parents)
        self._rel(labels, S.unnamed)
        for nabels_type, nabels in named_nabels:
            self._rel(nabels_type, nabels_type)
        for node_var in self._node_vars.values():
            if not node_var.is_constant:
                node_var.vals = None

    def _rel(self, from_nabels: Nabels, nabels_type: str):
        to_nabels = self._node_vars.children.vals
        config = self._nabels_configs[nabels_type]
        rel_args = tuple(reversed((from_nabels, to_nabels))) if config.reversed else from_nabels, to_nabels
        from_nabel: N
        to_nabel: N
        for from_nabel, to_nabel in product(*rel_args):
            if config.rel:
                config.rel(from_nabel, to_nabel)
            if config.name_as_label:
                to_nabel.node.update_labels(from_nabel.name)
            if config.labels_inherit:
                to_nabel.node.update_labels(from_nabel.node.labels)

    def __mul__(self, other: N_potcol):
        for node_var_type, node_var in self._node_vars.items():
            if not node_var.is_constant and not node_var.vals:
                node_var.vals = other
                break
        if all(map(bool, self._node_vars.values())):
            return self()
        return self


class N(HasName):
    def __init__(self, name: str, *labels: Nabel, relationer: Relationer = None, **named_nabels):
        super().__init__()
        self.node = Node(name=name)
        self.children: list = []
        self.relationer = relationer
        setattr(NN, name, self)

        if (labels or named_nabels) and self.relationer:
            self.relationer(None, self, *labels, **named_nabels)

    @property
    def name(self) -> str:
        return str(self.node[S.name])

    def get_node(self) -> Node:
        return self.node

    def __call__(self, name: str, *labels: str | N, **kwargs) -> N:
        name = name or '-'.join(map(lambda l: l if isinstance(l, str) else l.name, labels))
        n = N(name=name, relationer=self.relationer, **kwargs)
        self.relationer(self, n, *labels, **kwargs)
        self.children.append(n)
        return n

    def __repr__(self):
        return f'N-{self.name}'

RR._map_func = R.get_children
NN._map_func = N.get_node

