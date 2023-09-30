from __future__ import annotations

import operator as op
import random
from dataclasses import dataclass, field
from functools import reduce
from itertools import product, tee
from typing import Callable, Iterable, Type, Tuple, Any

from more_itertools import bucket, unique_everseen
from py2neo import Graph, Relationship, Node

from src.utils import save_iterabilize, DictClass, col_to_str, save_flatten

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
    def keys(cls):
        return cls.dir().keys()

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
            val.children = set()

    @classmethod
    def is_name_free(cls, name: str) -> bool:
        return name not in cls.dir()

    @classmethod
    def get_free_name(cls, name: str = '') -> str:
        new_name = name
        while not cls.is_name_free(new_name):
            new_name = name + f'--{random.random()}'
        return new_name

    @classmethod
    def set(cls, name: str, obj):
        if not cls.is_name_free(name):
            raise ValueError
        setattr(cls, name, obj)

    @classmethod
    def clear(cls) -> None:
        to_removes = [key for key in cls.__dict__ if not key.startswith('_')]
        for to_remove in to_removes:
            delattr(cls, to_remove)


# TODO: think if should store by name or rather by some id
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
    def __init__(self, name: str | Relationship, change_name=False):
        rel_name = self.extract_name_from_relationship(name) if str(name).startswith("<class 'py2neo.data.") else name if isinstance(name, str) else None
        if not change_name and not RR.is_name_free(rel_name):
            raise ValueError
        elif change_name:
            rel_name = RR.get_free_name(rel_name)

        super().__init__(name=rel_name)
        self.rel: Type[Relationship] = Relationship.type(rel_name)
        self.children: set[Relationship] = set()
        RR.set(rel_name, self)
        # setattr(RR, rel_name, self)

    @property
    def name(self) -> str:
        return self._name

    def get_rel(self) -> Type[Relationship]:
        return self.rel

    def get_children(self) -> set[Relationship]:
        return self.children

    @classmethod
    def extract_name_from_relationship(cls, r: Relationship) -> str:
        return str(r).removeprefix("<class 'py2neo.data.").removesuffix("'>")

    def __call__(self, *args, **kwargs) -> Relationship:
        args = tuple(map(lambda a: a.node if isinstance(a, N) else a, args))
        r = self.rel(*args, **kwargs)
        self.children.add(r)
        return r

    def __repr__(self):
        return f'R-{self.name}'


class N(HasName):
    pass


N_potcol = Tuple[N] | N
Nabel = N | str  # node/label
Nabels = Iterable[Nabel]
NabelSome = Nabel | Nabels
NodeSome = Node | Iterable[Node]


@dataclass
class NodeVar:
    _values: Nabels = field(default=tuple())
    is_constant: bool = field(default=False)
    vals = property(
        lambda self: self._values,
        lambda self, val: setattr(self, '_values', save_iterabilize(val)) if not self.is_constant else None
    )

    def __post_init__(self):
        self.vals = self._values


@dataclass
class NodeVars(DictClass):
    parents: NodeVar  = field(default_factory=lambda: NodeVar())
    children: NodeVar = field(default_factory=lambda: NodeVar())
    unnamed: NodeVar  = field(default_factory=lambda: NodeVar())


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

    @classmethod
    def filter_reversed(cls, to_filters: Iterable[str], kwargs: dict | bool = None) -> Iterable[str] | dict[str, Any]:
        return cls.filter_suffix(to_filters, S.reversed, kwargs)


@dataclass
class NabelConfig(DictClass):
    rel: R = field(default=None)
    name_as_label: bool = field(default=True)
    labels_inherit: bool = field(default=False)
    reversed: bool = field(default=False)

    @classmethod
    def make_from(cls, name: str, kwargs: dict, change_name=False, **defaults):
        params = {param_name.removeprefix(f'{name}_'): param_value for param_name, param_value in kwargs.items() if name in param_name and param_value is not None}
        if S.rel in params and not isinstance(params[S.rel], (R, type(None))):
            params[S.rel] = R(params[S.rel], change_name=change_name)
        for default_key, default_value in defaults.items():
            if default_key not in params or (S.rel not in default_key and params[default_key] is None):
                params[default_key] = default_value
        return NabelConfig(**params)

    @classmethod
    def extract_nabels_configs(cls, kwargs: dict, change_name: bool) -> dict[str, NabelConfig]:
        kwargs = cls._adjust_kwargs(kwargs)
        return {
            **cls._extract_parents_nabels_configs(kwargs, change_name),
            **cls._extract_unnamed_nabels_configs(kwargs, change_name),
            **cls._extract_named_nabels_configs(kwargs, change_name),
        }

    @classmethod
    def _adjust_kwargs(cls, kwargs: dict) -> dict:
        for key in (S.rel, S.labels_inherit, S.name_as_label):
            if key in kwargs:
                kwargs[f'{S.parents}_{key}'] = kwargs[key]
                del kwargs[key]
        return kwargs

    @classmethod
    def _extract_parents_nabels_configs(cls, kwargs, change_name: bool) -> dict[str, NabelConfig]:
        return {S.parents: cls.make_from(S.parents, kwargs, change_name=change_name, labels_inherit=True)}

    @classmethod
    def _extract_unnamed_nabels_configs(cls, kwargs: dict, change_name: bool) -> dict[str, NabelConfig]:
        return {S.unnamed: cls.make_from(S.unnamed, kwargs, change_name=change_name)}

    @classmethod
    def _extract_named_nabels_configs(cls, kwargs: dict, change_name: bool) -> dict[str, NabelConfig]:
        permissable = filter(cls.map_to_contained_key, kwargs.keys())
        not_parents = filter(lambda key: not key.startswith(S.parents), permissable)
        not_unnamed = filter(lambda key: not key.startswith(S.unnamed), not_parents)
        named = map(lambda key: key.removesuffix(f'_{cls.map_to_contained_key(key)}'), not_unnamed)
        unique = unique_everseen(named)
        unique_keys, unique_to_map = tee(unique)
        unique_confs = map(lambda unique_key: cls.make_from(unique_key, kwargs, change_name=change_name), unique_to_map)
        named_nabels_configs = dict(zip(unique_keys, unique_confs))
        return named_nabels_configs

    def __bool__(self):
        return reduce(op.or_, map(bool, (self.rel, self.name_as_label, self.labels_inherit)), False)


class Relationer:
    '''
        [NAME]_rel, [NAME]_labels_inherit, [NAME]_name_as_label
    '''
    def __init__(self, *labels, change_name=False, **kwargs):
        super().__init__()
        self._n_call = 0
        self._is_reversed_used = False
        self._node_vars = NodeVars()
        self._nabels_configs: dict[str, NabelConfig] = NabelConfig.extract_nabels_configs(kwargs, change_name)

    def __call__(self, parents: NabelSome, children: NabelSome, *unnamed: Nabel, **named_nabels: NabelSome) -> Relationer:
        named_nabels[S.parents] = parents
        named_nabels[S.children] = children
        named_nabels[S.unnamed] = unnamed
        self._set_nabels(named_nabels)
        self._rel_node_vars()
        self._reverse_if_needed()
        self._node_vars = NodeVars()
        return self

    def get_confs(self, name: str) -> NabelConfig:
        return self._nabels_configs[name]

    def get_rel(self, name: str) -> R:
        return self.get_confs(name).rel

    def _set_nabels(self, nabels_dict: dict[str, NabelSome]):
        for nabels_type, nabels in nabels_dict.items():
            self._node_vars[nabels_type] = NodeVar()
            self._node_vars[nabels_type].vals = save_flatten(nabels, list, map_func=to_node)

    def _rel_node_vars(self) -> None:
        for nabels_type, node_var in self._node_vars.items():
            self._rel(node_var.vals, nabels_type)

    def _reverse_if_needed(self) -> None:
        if self._is_reversed_used and self._n_call == 0:
            self._n_call += 1
            self.__call__(
                self._node_vars.parents.vals,
                self._node_vars.children.vals,
                *self._node_vars.unnamed.vals,
                **{name: nabels.vals for name, nabels in self._node_vars.items() if name not in (S.parents, S.children, S.unnamed)}
            )
        else:
            self._is_reversed_used = False
            self._n_call = 0

    def _rel(self, from_nabels: Iterable[Node], nabels_type: str):
        if nabels_type == S.children:
            return
        to_nabels: Nabels = self._node_vars.children.vals
        from_nabels, to_nabels = self._get_reversed_arguments_if_needed(from_nabels, to_nabels, nabels_type)
        config = self._nabels_configs[nabels_type]
        self._is_reversed_used |= config.reversed
        for from_nabel, to_nabel in product(from_nabels, to_nabels):
            if config.rel:
                config.rel(from_nabel, to_nabel)
            if config.name_as_label:
                to_nabel.update_labels([from_nabel[S.name]])
            if config.labels_inherit:
                to_nabel.update_labels(from_nabel.labels)

    def _get_reversed_arguments_if_needed(self, from_nabels: NodeSome, to_nabels: NodeSome, nabels_type: str) -> tuple:
        return (to_nabels, from_nabels) if self._nabels_configs[nabels_type].reversed else (from_nabels, to_nabels)

    def __mul__(self, other: N_potcol):
        for node_var_type, node_var in self._node_vars.items():
            if not node_var.is_constant and not node_var.vals:
                node_var.vals = other
                break
        if all(map(bool, self._node_vars.values())):
            return self()
        return self


class N(HasName, dict):  # TODO dict was added for complience with save_iterazable, check if needed

    _common_relationer = None
    # @classmethod
    # def set_with_id(cls, state: bool = True):
    #     cls.with_id = state

    @classmethod
    def set_common_relationer(cls, relationer: Relationer):
        cls._common_relationer = relationer

    def __init__(self, name: str, *labels: Nabel, relationer: Relationer = None, **named_nabels):
        super().__init__()
        self.node = Node(name=name)
        self.children: set = set()
        self.relationer = relationer or self._common_relationer
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

    def get_labels(self) -> set[str]:
        return self.labels

    def __call__(self, name: str, *labels: str | N, **kwargs) -> N:
        name = name or '-'.join(map(lambda l: l if isinstance(l, str) else l.name, labels))
        n = N(name=name, relationer=self.relationer, **kwargs)
        self.relationer(self, n, *labels, **kwargs)
        self.children.add(n)
        return n

    def __getitem__(self, item) -> Any:
        return self.node[item]

    def __setitem__(self, key, value) -> None:
        self.node[key] = value

    def __repr__(self):
        return f'N-{self.name}'

    def keys(self) -> Iterable:
        return self.node.keys()


class ChildMaker:
    def __init__(self, func: Callable[[Any], Any], rel: Relationer = None):
        self.func = func
        self.rel = rel

    def __call__(self, parents, *args, **kwargs):
        parents = save_iterabilize(parents, tuple)
        children = self.func(parents)
        if self.rel:
            self.rel(parents, children, *args, **kwargs)
        return children


def to_node(elem):
    return elem.node if isinstance(elem, N) else elem

RR._map_func = R.get_children
NN._map_func = N.get_node

