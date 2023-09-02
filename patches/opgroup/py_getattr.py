# Implements optimizations for various py__getattribute__ methods.

from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.klass import ClassValue

from itertools import repeat
from operator import attrgetter

from ..common import get_cached_scope_definitions, state, Values, starchain


rep_state = repeat(state)


def apply():
    optimize_ClassValue_py__getattribute__()


def py__getattribute__(self: ClassValue, name_or_str, name_context=None, position=None, analysis_errors=True):
    if name_or_str.__class__ is not str:
        name_or_str = name_or_str.value
    names = get_cached_scope_definitions(self.tree_node.children[-1])[name_or_str]
    results = map(tree_name_to_values, rep_state, repeat(self.as_context()), names)
    return Values(starchain(results))


def optimize_ClassValue_py__getattribute__():
    ClassValue.py__getattribute__ = py__getattribute__
