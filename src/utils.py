from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Type, Callable, Any, Generator, Collection

from py2neo import Node


###########
# General #
###########


def save_iterabilize(iterable: Iterable | None, default: Type | Callable = list) -> Iterable | Collection:
    iterable: Iterable = iterable or default()
    if not isinstance(iterable, Iterable) or isinstance(iterable, str):
        if isinstance(iterable, str):
            iterable = (iterable, )
        iterable = default(iterable)
    return iterable


def reapply(fn, arg, n=None, until=None, as_long=None):
    if sum([arg is None for arg in (n, until, as_long)]) < 2:
        raise ValueError
    cond = (lambda a: n > 0) if n is not None else (lambda a: not until(a)) if until is not None else as_long if as_long is not None else (lambda a: False)
    if isinstance(arg, Iterable) and not isinstance(arg, str):
        arg = list(arg)
    while cond(arg):
        arg = fn(arg)
    return arg


class DictClass:
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        print('To verify: ',  self.__dict__)
        return self.__dict__.__setitem__(key, value)

    dict: Callable[[], dict] = asdict
    values: Callable[[], Any] = lambda self: self.dict().values()
    keys: Callable[[], Any] = lambda self: self.dict().keys()
    items: Callable[[], Any] = lambda self: self.dict().items()

    @classmethod
    def map_to_contained_key(cls, k) -> str | None:
        return next(filter(k.__contains__, cls().keys()), None)


############
# Concrete #
############

def col_to_str(col):
    if isinstance(col, (tuple, list, Generator, map, filter)):
        return (col_to_str(c) for c in col)
    if isinstance(col, str):
        return col
    if hasattr(col, 'name'):
        return col.name
    if isinstance(col, Node):
        return col['name']
    return str(col)
