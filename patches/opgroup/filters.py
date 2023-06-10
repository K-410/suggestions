# This module implements optimizations for various filter types.

def apply():
    optimize_SelfAttributeFilter_values()


# Optimizes SelfAttributeFilter.values to search within the class scope.
def optimize_SelfAttributeFilter_values():
    from jedi.inference.value.instance import SelfAttributeFilter
    from ..tools import is_basenode, is_namenode, is_funcdef, is_classdef
    from itertools import repeat

    def values(self: SelfAttributeFilter):
        names   = []
        scope   = self._parser_scope
        context = self.parent_context

        # We only care about classes, and stubs never have self definitions.
        if is_classdef(scope) and not context.is_stub():
            with any.measure_total:
                class_nodes = scope.children[-1].children
                pool = []

                for n in filter(is_funcdef, class_nodes):
                    pool += n.children[-1].children

                # Recurse node trees.
                for n in filter(is_basenode, pool):
                    pool += n.children

                # Get name definitions.
                for n in filter(is_namenode, pool):
                    if n.get_definition(include_setitem=True):
                        names += [n]

            names = list(self._filter(names))
            names = list(map(self.name_class, repeat(context), names))
        return names

    SelfAttributeFilter.values = values

