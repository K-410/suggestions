# Implements optimizations for various py__getattribute__ methods.

from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.klass import ClassValue, ValueSet
from jedi.inference.base_value import HelperValueMixin

from itertools import repeat, compress
from functools import reduce
from operator import attrgetter, eq, or_

from ..common import get_scope_name_definitions, state


rep_state = repeat(state)
get_value = attrgetter("value")
get_set   = attrgetter("_set")


def apply():
    optimize_ClassValue_py__getattribute__()


def py__getattribute__(self: ClassValue, name_or_str, name_context=None, position=None, analysis_errors=True):
    if name_or_str.__class__ is str:
        names = get_scope_name_definitions(self.tree_node.children[-1])
        names = compress(names, map(eq, repeat(name_or_str), map(get_value, names)))
        results = map(tree_name_to_values, rep_state, repeat(self.as_context()), names)

        if values := list(results):
            return ValueSet(reduce(or_, map(get_set, values)))
        else:
            print("py_getattr failed to find:", name_or_str, f"({self})")

    else:
        # XXX: Not handled. Should we handle this? How often is this called?
        print("py_getattr called with Name (not name_str):", name_or_str)

    return HelperValueMixin.py__getattribute__(self, name_or_str, name_context, position, analysis_errors)


def optimize_ClassValue_py__getattribute__():
    ClassValue.py__getattribute__ = py__getattribute__
