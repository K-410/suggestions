# Optimizes ``_defined_names`` and methods that call it .

from textension.utils import _patch_function


def apply():
    optimize_defined_names()
    optimize_ExprStmt_get_defined_names()
    optimize_ForStmt_and_SyncCompFor_get_defined_names()
    optimize_WithStmt_get_defined_names()
    optimize_WithStmt_get_defined_names()
    optimize_NamedExpr_get_defined_names()


def optimize_defined_names():
    from parso.python import tree
    from builtins import iter

    PythonNode_types = {'testlist_star_expr', 'testlist_comp', 'exprlist',
                        'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}

    subset  = {"testlist_star_expr", "testlist_comp", "exprlist", "testlist", "atom", "star_expr"}
    subset1 = {"testlist_star_expr", "testlist_comp", "exprlist", "testlist"}

    def _defined_names(node, include_setitem):
        pool = [node]

        for node in iter(pool):
            node_type = node.type
            if node_type in PythonNode_types:
                if node_type in subset:
                    if node_type in subset1:
                        pool += node.children[::2]

                    # Must be one in {"atom", "star_expr"}
                    else:
                        pool += [node.children[1]]
                else:
                    trailer_children = node.children[-1].children
                    value = trailer_children[0].value

                    if value is ".":  # String comparisons use direct check, skip overload.
                        yield trailer_children[1]

                    elif value is "[" and include_setitem:
                        for child in node.children[-2::-1]:
                            child_type = child.type
                            if child_type == 'trailer':
                                yield trailer_children[1]
                                break
                            elif child_type == 'name':
                                yield child
                                break
            # Is a Name node.
            else:
                yield node

        return

    _patch_function(tree._defined_names, _defined_names, rename=False)


# ForStmt and SyncCompFor use the same approach.
def optimize_ForStmt_and_SyncCompFor_get_defined_names():
    from parso.python.tree import ForStmt, SyncCompFor, _defined_names

    PythonNode_types = {'testlist_star_expr', 'testlist_comp', 'exprlist',
                        'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}

    def get_defined_names(self, include_setitem=False):
        node = self.children[1]
        if node.type in PythonNode_types:
            return _defined_names(node, include_setitem)
        yield node

    ForStmt.get_defined_names     = get_defined_names
    SyncCompFor.get_defined_names = get_defined_names


def optimize_WithStmt_get_defined_names():
    from parso.python.tree import WithStmt, _defined_names

    PythonNode_types = {'testlist_star_expr', 'testlist_comp', 'exprlist',
                        'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}

    def get_defined_names(self, include_setitem=False):
        for with_item in self.children[1:-2:2]:
            # Check with items for 'as' names.
            if with_item.type == 'with_item':
                node = with_item.children[2]
                if node in PythonNode_types:
                    return _defined_names(node, include_setitem)
                yield node

    WithStmt.get_defined_names = get_defined_names


def optimize_KeywordStatement_get_defined_names():
    from parso.python.tree import KeywordStatement, _defined_names

    PythonNode_types = {'testlist_star_expr', 'testlist_comp', 'exprlist',
                        'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}

    def get_defined_names(self, include_setitem=False):
        keyword = self.keyword
        if keyword == 'del':
            node = self.children[1]
            if node in PythonNode_types:
                return _defined_names(node, include_setitem)

        elif keyword in ('global', 'nonlocal'):
            yield from self.children[1::2]

        else:
            yield ()

    KeywordStatement.get_defined_names = get_defined_names


def optimize_NamedExpr_get_defined_names():
    from parso.python.tree import NamedExpr, _defined_names

    PythonNode_types = {'testlist_star_expr', 'testlist_comp', 'exprlist',
                        'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}

    def get_defined_names(self, include_setitem=False):
        node = self.children[0]
        if node in PythonNode_types:
            return _defined_names(node, include_setitem)
        yield node

    NamedExpr.get_defined_names = get_defined_names


def optimize_ExprStmt_get_defined_names():
    from parso.python.tree import ExprStmt, _defined_names

    len = list.__len__
    def get_defined_names(self, include_setitem=False):
        children = self.children

        if children[1].type == "annassign":
            # ``children[0]`` is the only definition of an annotation.
            yield children[0]

        i = 1
        for c in children[1:len(children) - 1:2]:

            # ``c`` is an assignment operator.
            if c.type == "operator" and c.value[-1] is "=":
                name = children[i - 1]
                if name.type not in {'testlist_star_expr', 'testlist_comp', 'exprlist',
                                     'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}:
                    yield name
                else:
                    yield from _defined_names(name, include_setitem)
            i += 2

    ExprStmt.get_defined_names = get_defined_names
