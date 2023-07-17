from __future__ import annotations

import operator as op
from dataclasses import dataclass, field
from functools import reduce
from itertools import product
from typing import Callable, Iterable, Type, Tuple, Any

from more_itertools import bucket, flatten
from py2neo import Graph, Relationship, Node

from src.utils import save_iterabilize, DictClass, col_to_str



Graph.create_all = lambda self, *subgraphs: [self.create(subgraph) for subgraph in subgraphs] is None and None


##################
# Implementation #
##################

@dataclass(frozen=True)
class Strings:
    reversed = 'reversed'
    name = 'name'
    children = 'children'
    parents = 'parents'
    unnamed = 'unnamed'
    kwargs = 'kwargs'
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

    @classmethod
    def get(cls, key, default=None):
        return cls.dir().get(key, default)

    @classmethod
    def clean_children(cls):
        for val in cls.vals():
            val.children = []


class NN(D): pass
class RR(D): pass


class HasName:
    def __init__(self, name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name: str = name

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
        return self._name

    def get_rel(self) -> Type[Relationship]:
        return self.rel

    def get_children(self) -> list[Relationship]:
        return self.children

    def __call__(self, *args, **kwargs) -> Relationship:
        r = self.rel(*tuple(map(lambda a: a.node if isinstance(a, N) else a, args)), **kwargs)
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
        if args and not self.is_constant:
            self.vals = save_iterabilize(args[0])
            return args[1:]
        return args

    def __post_init__(self):
        self.vals = self._values


@dataclass
class NodeVars(DictClass):
    parents: NodeVar  = field(default_factory=lambda: NodeVar(None, False))
    children: NodeVar = field(default_factory=lambda: NodeVar(None, False))


class Filterer:

    @classmethod
    def filter_suffix(cls, to_filters: Iterable[str], prefix: str, kwargs: dict | bool = None) -> Iterable[str] | dict[str, Any]:
        if kwargs is True:
            kwargs = to_filters
        with_prefixes = filter(lambda elem: elem.endswith(prefix), to_filters)
        return {k:kwargs[k] for k in with_prefixes} if kwargs is not None else with_prefixes

    @classmethod
    def filter_rel(cls, to_filters: Iterable[str], kwargs: dict | bool = None) -> Iterable[str]| dict[str, Any]:
        return cls.filter_suffix(to_filters, S.rel, kwargs)

    @classmethod
    def filter_name_as_labels(cls, to_filters: Iterable[str], kwargs: dict | bool = None) -> Iterable[str] | dict[str, Any]:
        return cls.filter_suffix(to_filters, S.name_as_label, kwargs)

    @classmethod
    def filter_labels_inherit(cls, to_filters: Iterable[str], kwargs: dict | bool = None) -> Iterable[str] | dict[str, Any]:
        return cls.filter_suffix(to_filters, S.labels_inherit, kwargs)


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

    def __bool__(self):
        return reduce(op.or_, map(bool, (self.rel, self.name_as_label, self.labels_inherit)), False)


class Relationer:
    '''
        [NAME]_rel, [NAME]_labels_inherit, [NAME]_name_as_label
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self._node_vars = NodeVars()
        for key in (S.rel, S.labels_inherit, S.name_as_label):
            if key in kwargs:
                kwargs[f'{S.parents}_{key}'] = kwargs[key]
                del kwargs[key]
        self._nabels_configs: dict[str, NabelConfig] = {
            S.parents: NabelConfig.make_from(S.parents, kwargs, labels_inherit=True),
            S.unnamed: NabelConfig.make_from(S.unnamed, kwargs),
        }
        truncated_keys = (key.removesuffix(f'_{NabelConfig.map_to_contained_key(key)}') for key in kwargs.keys() if NabelConfig.map_to_contained_key(key) is not None and not any((key.startswith(prefix) for prefix in (S.parents, S.unnamed))))
        unique_keys = set(truncated_keys)
        for unique_key in unique_keys:
            self._nabels_configs[unique_key] = NabelConfig.make_from(unique_key, kwargs)

    def __call__(self, *labels, **named_nabels: Nabel):
        labels = list(map(lambda l: to_node(l, 2), labels))
        labels = self._node_vars.parents.set_first_from(labels)
        labels = self._node_vars.children.set_first_from(labels)
        self._rel(self._node_vars.parents.vals, S.parents)
        labels = list(l for label in filter(bool, labels) for l in (label if isinstance(label, (tuple, list)) else (label, )))
        self._rel(labels, S.unnamed)
        for nabels_type, nabels in named_nabels.items():
            self._rel(save_iterabilize(nabels), nabels_type)
        for node_var in self._node_vars.values():
            if not node_var['is_constant']:
                node_var['vals'] = None

    def _rel(self, from_nabels: Iterable[Node], nabels_type: str):
        to_nabels: Nabels = self._node_vars.children.vals
        config = self._nabels_configs[nabels_type]
        rel_args = tuple(reversed((from_nabels, to_nabels))) if config.reversed else from_nabels, to_nabels
        if not bool:
            return
        for from_nabel, to_nabel in product(*rel_args):
            if config.rel:
                config.rel(from_nabel, to_nabel)
            if config.name_as_label:
                to_nabel.update_labels(from_nabel[S.name])
            if config.labels_inherit:
                to_nabel.update_labels(from_nabel.labels)

    def __mul__(self, other: N_potcol):
        for node_var_type, node_var in self._node_vars.items():
            if not node_var.is_constant and not node_var.vals:
                node_var.vals = other
                break
        if all(map(bool, self._node_vars.values())):
            return self()
        return self


class N(HasName, dict):  # TODO dict was added for complience with save_iterazable, check if needed
    def __init__(self, name: str, *labels: Nabel, relationer: Relationer = None, **named_nabels):
        super().__init__()
        self.node = Node(name=name)
        self.children: list = []
        self.relationer = relationer
        setattr(NN, name, self)

        bucketed = bucket(named_nabels, key=lambda nl: NabelConfig.map_to_contained_key(nl) is None)
        relationer_nabels = {key: named_nabels[key] for key in bucketed[False]}
        node_nabels = {key: named_nabels[key] for key in bucketed[True]}

        self.add_kwargs(node_nabels)
        if relationer:
            self.relationer(None, self, *labels, **relationer_nabels)
        else:
            self.node.update_labels(list(col_to_str(labels)))

    def add_kwargs(self, kwargs: dict) -> None:
        for key, val in kwargs.items():
            self.node[key] = val

    @property
    def name(self) -> str:
        return str(self.node[S.name])

    def get_node(self) -> Node:
        return self.node

    @property
    def labels(self) -> set[str]:
        return self.node.labels

    def get_labes(self) -> set[str]:
        return self.labels

    def __call__(self, name: str, *labels: str | N, **kwargs) -> N:
        name = name or '-'.join(map(lambda l: l if isinstance(l, str) else l.name, labels))
        n = N(name=name, relationer=self.relationer, **kwargs)
        self.relationer(self, n, *labels, **kwargs)
        self.children.append(n)
        return n

    def __getitem__(self, item) -> Any:
        return self.node[item]

    def __setitem__(self, key, value) -> None:
        self.node[key] = value

    def __repr__(self):
        return f'N-{self.name}'

    def keys(self) -> Iterable:
        return self.node.keys()


def to_node(elem, n):
    if n == 0:
        return elem
    return elem.node if isinstance(elem, N) else list(map(lambda e: to_node(e, n-1), elem)) if isinstance(elem, (list, tuple)) else elem

RR._map_func = R.get_children
NN._map_func = N.get_node

