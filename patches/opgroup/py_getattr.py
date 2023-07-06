# Implements optimizations for various py__getattribute__ methods.

from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.klass import ClassValue, ValueSet
from jedi.inference.base_value import HelperValueMixin

from itertools import repeat, compress
from operator import attrgetter, eq, or_
from functools import reduce

from ..common import get_scope_name_definitions, state


rep_state = repeat(state)
get_value = attrgetter("value")
get_set   = attrgetter("_set")


def merge_value_sets(value_sets):
    values = object.__new__(ValueSet)
    values._set = reduce(or_, map(get_set, value_sets))
    return values


def apply():
    pass
    optimize_ClassValue_py__getattribute__()


def find_member_by_string(name_str: str, value: ClassValue):
    names = get_scope_name_definitions(value.tree_node.children[-1])
    names = compress(names, map(eq, repeat(name_str), map(get_value, names)))
    results = map(tree_name_to_values, rep_state, repeat(value.as_context()), names)
    return merge_value_sets(results)


def py__getattribute__(self: ClassValue, name_or_str, name_context=None, position=None, analysis_errors=True):
    if name_or_str.__class__ is str:
        if values := find_member_by_string(name_or_str, self):
            print("py_getattr found values for:", name_or_str)
            return values
        else:
            print("py_getattr failed to find:", name_or_str)
    else:
        # XXX: Not handled. Should we handle this? How often is this called?
        print("py_getattr called with Name (not name_str):", name_or_str)
    return HelperValueMixin.py__getattribute__(
        self, name_or_str, name_context, position, analysis_errors)


def optimize_ClassValue_py__getattribute__():
    ClassValue.py__getattribute__ = py__getattribute__
