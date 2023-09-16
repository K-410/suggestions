# Implements optimizations for various py__getattribute__ methods.

from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.klass import ClassValue

from itertools import repeat
from functools import partial

from ..common import get_cached_scope_definitions, state, Values, starchain
from textension.utils import inline


rep_state = repeat(state)


def apply():
    optimize_ClassValue_py__getattribute__()
    optimize_AbstractContext_py__getattribute__()
    optimize_HelperValueMixin_py__getattribute__()


def optimize_ClassValue_py__getattribute__():

    def py__getattribute__(self: ClassValue,
                        name_or_str,
                        name_context=None,
                        position=None,
                        analysis_errors=True):

        if name_or_str.__class__ is not str:
            name_or_str = name_or_str.value
        names = get_cached_scope_definitions(self.tree_node.children[-1])[name_or_str]
        results = map(tree_name_to_values, rep_state, repeat(self.as_context()), names)
        return Values(starchain(results))

    ClassValue.py__getattribute__ = py__getattribute__


# Strip to barebone. Inline goto. Also removes predefined names.
def optimize_AbstractContext_py__getattribute__():
    from jedi.inference.analysis import add as add_analysis
    from jedi.inference.context import AbstractContext, _get_global_filters_for_name
    from jedi.inference.finder import filter_name

    from textension.utils import starchain
    from ..common import Values, is_namenode, map_infer

    def py__getattribute__(self: AbstractContext,
                           name_or_str,
                           name_context=None,
                           position=None,
                           analysis_errors=True):

        filters = _get_global_filters_for_name(
            self, name_or_str if is_namenode(name_or_str) else None, position)
        if names := filter_name(filters, name_or_str):
            # XXX: Can we do this?
            # if values := Values(starchain(map_infer(names))):
            if values := Values(next(map_infer(names), ())):
                return values

        if analysis_errors and not names and is_namenode(name_or_str):
            message = ("NameError: name '%s' is not defined." % str(name_or_str))
            add_analysis(name_context or self, 'name-error', name_or_str, message)
        return self._check_for_additional_knowledge(name_or_str, name_context or self, position)

    AbstractContext.py__getattribute__ = py__getattribute__


def optimize_HelperValueMixin_py__getattribute__():
    from jedi.inference.base_value import HelperValueMixin
    from jedi.inference.analysis import add_attribute_error
    from parso.python.tree import Name
    from textension.utils import starchain
    from operator import methodcaller
    from ..common import Values, NO_VALUES

    @inline
    def map_infer(names):
        return partial(map, methodcaller("infer"))
    
    def py__getattribute__(self: HelperValueMixin,
                           name_or_str,
                           name_context=None,
                           position=None,
                           analysis_errors=True):

        if names := self.goto(name_or_str, name_context or self, analysis_errors):
            if values := Values(starchain(map_infer(names))):
                return values

        elif analysis_errors and isinstance(name_or_str, Name):
            add_attribute_error(name_context or self, self, name_or_str)
        return NO_VALUES

    HelperValueMixin.py__getattribute__ = py__getattribute__
