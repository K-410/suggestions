# This module implements safe optimizations for jedi.
# A safe optimization is independent and does not modify behavior.

from textension.utils import (
    _forwarder,
    _unbound_getter,
    _unbound_method,
    _unbound_attrcaller,
    _patch_function,
    truthy_noargs,
    falsy,
    falsy_noargs)


from itertools import repeat
from jedi.inference.base_value import NO_VALUES


def apply():
    optimize_Value_methods()
    optimize_ImportFrom_get_defined_names()
    optimize_create_stub_map()
    optimize_StringComparisonMixin__eq__()
    optimize_ValueContext()
    optimize_ClassMixin()
    optimize_Compiled_methods()
    optimize_CompiledName()
    optimize_TreeContextMixin()
    optimize_MergedFilter()
    optimize_shadowed_dict()
    optimize_static_getmro()
    optimize_AbstractTreeName_properties()
    optimize_BaseName_properties()
    optimize_Leaf()
    optimize_split_lines()
    optimize_Param_get_defined_names()
    optimize_platform_system()
    optimize_getmodulename()
    optimize_is_big_annoying_library()
    optimize_iter_module_names()
    optimize_Context_methods()
    optimize_CompiledIntance_init()

    optimize_Node_get_previous_leaf()
    optimize_ValueContext_methods()
    optimize_ValueSet_methods()
    optimize_getattr_static()
    optimize_BaseNode_get_leaf_for_position()
    optimize_remove_del_stmt()
    optimize_infer_node_if_inferred()
    optimize_try_to_load_stub_cached()
    optimize_imports_iter_module_names()
    optimize_BaseNode_end_pos()
    optimize_Leaf_end_pos()
    optimize_CompiledValue_api_type()
    optimize_BaseNode_get_last_leaf()
    optimize_pickling()
    optimize_LazyTreeValue_infer()
    optimize_LazyInstanceClassName()
    optimize_get_module_info()
    optimize_tree_name_to_values()
    optimize_infer_expr_stmt()
    optimize_ClassMixin_py__mro__()
    optimize_Param_name()
    optimize_CompiledInstanceName()
    optimize_AbstractContext_get_root_context()
    optimize_NodesTree_copy_nodes()
    optimize_NodeOrLeaf_get_next_leaf()
    optimize_find_overload_functions()
    optimize_builtin_from_name()


rep_NO_VALUES = repeat(NO_VALUES).__next__


def optimize_imports_iter_module_names():
    from jedi.inference.imports import ImportName
    from jedi.inference import imports
    from jedi.inference.compiled.subprocess.functions import get_builtin_module_names, _iter_module_names
    from textension.utils import _named_index
    from itertools import chain

    def iter_module_names(inference_state, module_context, search_path,
                          module_cls=ImportName, add_builtin_modules=True):
        
        class module_name(module_cls, tuple):
            __init__ = tuple.__init__
            _from_module_context = module_context
            string_name = _named_index(0)

        ret = _iter_module_names(None, search_path)
        # add builtin module names
        if add_builtin_modules:
            ret = chain(get_builtin_module_names(None), ret)
        return list(map(module_name, zip(ret)))

    _patch_function(imports.iter_module_names, iter_module_names)


# Removes sys_path argument when trying to load stubs. It's just terrible.
def optimize_try_to_load_stub_cached():
    from jedi.inference.gradual.typeshed import try_to_load_stub_cached, _try_to_load_stub
    from ..tools import state

    stub_module_cache = state.stub_module_cache

    def try_to_load_stub_cached_o(inference_state, import_names, *args, **kwargs):
        assert import_names, f"Why even pass invalid import_names? -> {repr(import_names)}"

        if import_names not in stub_module_cache:
            # Jedi will traverse sys.path and thrash the disk looking for
            # stub files that don't exist. We don't want that.
            if "sys_path" in kwargs:
                kwargs["sys_path"] = []
            else:
                args = args[:-1] + ([],)
            stub_module_cache[import_names] = _try_to_load_stub(inference_state, import_names, *args, **kwargs)
        return stub_module_cache[import_names]

    _patch_function(try_to_load_stub_cached, try_to_load_stub_cached_o)


def optimize_infer_node_if_inferred():
    from jedi.inference.syntax_tree import _infer_node, infer_node
    from jedi.inference import syntax_tree
    from ..common import state, state_cache

    cache = state.memoize_cache

    # The actual ``_infer_node`` function.
    _infer_node = _infer_node.__closure__[0].cell_contents.__closure__[0].cell_contents

    @state_cache
    def _infer_node_if_inferred(context, element):
        # Unlikely.
        if predefined_names := context.predefined_names:
            parent = element
            while parent := parent.parent:
                if predefined_names.get(parent):
                    return _infer_node(context, element)
                
        key = context, element
        if key in memo:
            return memo[key]
        memo[key] = NO_VALUES
        memo[key] = ret = _infer_node(context, element)
        return ret

    memo = cache[_infer_node] = {}

    _patch_function(syntax_tree._infer_node_if_inferred, _infer_node_if_inferred)
    _patch_function(infer_node, _infer_node_if_inferred.__closure__[0].cell_contents)


def optimize_remove_del_stmt():
    from jedi.inference import finder

    def _remove_del_stmt(names):
        for name in names:
            if n := name.tree_name:
                if n.parent.type == "del_stmt":
                    continue
            yield name

    _patch_function(finder._remove_del_stmt, _remove_del_stmt)


# Optimize to use iterative (non-recursive) binary search.
def optimize_BaseNode_get_leaf_for_position():
    from parso.tree import BaseNode
    from builtins import len
    from ..common import node_types

    def get_leaf_for_position(self: BaseNode, position, include_prefixes=False):
        if position > self.children[-1].end_pos or position < (1, 0):
            raise ValueError(f"Position must be within the bounds of the node. "
                             f"(1, 0) < {position} <= {self.children[-1].end_pos}")

        while self.__class__ in node_types:
            lo = 0
            hi = len(children := self.children) - 1

            while lo < hi:
                i = (lo + hi) // 2
                if position > children[i].end_pos:
                    lo = i + 1
                    continue
                hi = i

            self = children[lo]
            if include_prefixes:
                continue

            elif position < self.start_pos:
                return None
        return self

    BaseNode.get_leaf_for_position = get_leaf_for_position


def optimize_ValueSet_methods():
    from jedi.inference.base_value import ValueSet

    del ValueSet.__bool__
    ValueSet.__eq__   = _forwarder("_set.__eq__")
    ValueSet.__hash__ = _forwarder("_set.__hash__")
    ValueSet.__iter__ = _forwarder("_set.__iter__")
    ValueSet.__len__  = _forwarder("_set.__len__")
    ValueSet.__ne__   = _forwarder("_set.__ne__")

    # Skip wrappers that aren't useful.
    def execute(self: ValueSet, args):
        if len(self._set) == 1:
            value, = self._set
            return value.py__call__(args)
        return ValueSet.from_sets(v.py__call__(args) for v in self._set)
    
    ValueSet.execute = execute


def optimize_ValueContext_methods():
    from jedi.inference.context import ValueContext
    from ..tools import state

    ValueContext.inference_state = state

    ValueContext.name            = _forwarder("_value.name")
    ValueContext.parent_context  = _forwarder("_value.parent_context")
    ValueContext.tree_node       = _forwarder("_value.tree_node")

    ValueContext.is_bound_method = _forwarder("_value.is_bound_method")
    ValueContext.is_class        = _forwarder("_value.is_class")
    ValueContext.is_compiled     = _forwarder("_value.is_compiled")
    ValueContext.is_instance     = _forwarder("_value.is_instance")
    ValueContext.is_module       = _forwarder("_value.is_module")
    ValueContext.is_stub         = _forwarder("_value.is_stub")
    ValueContext.py__doc__       = _forwarder("_value.py__doc__")
    ValueContext.py__name__      = _forwarder("_value.py__name__")

    ValueContext.get_qualified_names = _forwarder("_value.get_qualified_names")
    ValueContext.get_value       = _unbound_getter("_value")

    # We don't use predefined names.
    ValueContext.predefined_names = type(type.__dict__)({})

    def __init__(self: ValueContext, value):
        self._value = value

    ValueContext.__init__ = __init__


def optimize_Node_get_previous_leaf():
    from parso.tree import NodeOrLeaf
    from ..common import node_types

    def get_previous_leaf(self: NodeOrLeaf):
        while parent := self.parent:
            i = 0
            for c in parent.children:
                if c is self:
                    if i is not 0:
                        self = parent.children[i - 1]
                        while self.__class__ in node_types:
                            self = self.children[-1]
                        return self
                    break
                i += 1
            self = parent
        return None

    NodeOrLeaf.get_previous_leaf = get_previous_leaf


# This just inlines the nested super().__init__ calls.
def optimize_CompiledIntance_init():
    from jedi.inference.value.instance import CompiledInstance
    from ..tools import state

    CompiledInstance.inference_state = state

    def __init__(self: CompiledInstance, inference_state, parent_context, class_value, arguments):
        self.parent_context  = parent_context
        self.class_value = class_value
        self._arguments  = arguments

    CompiledInstance.__init__ = __init__


def optimize_Context_methods():
    from jedi.inference.context import ModuleContext, GlobalNameFilter

    ModuleContext.py__file__   = _forwarder("_value.py__file__")
    ModuleContext.string_names = _forwarder("_value.string_names")
    ModuleContext.code_lines   = _forwarder("_value.code_lines")
    ModuleContext.get_global_filter = _unbound_method(GlobalNameFilter)


# Optimizes various methods on Value that return a fixed value.
# This makes them faster and debug stepping in Jedi more bearable.
def optimize_Value_methods():
    from jedi.inference.base_value import Value

    # Automatically False.
    Value.is_class            = falsy_noargs
    Value.is_class_mixin      = falsy_noargs
    Value.is_instance         = falsy_noargs
    Value.is_function         = falsy_noargs
    Value.is_module           = falsy_noargs
    Value.is_namespace        = falsy_noargs
    Value.is_compiled         = falsy_noargs
    Value.is_bound_method     = falsy_noargs
    Value.is_builtins_module  = falsy_noargs
    Value.get_qualified_names = falsy_noargs

    Value.get_type_hint       = falsy
    Value.py__bool__          = truthy_noargs

    Value.is_stub = _forwarder("parent_context.is_stub")


def optimize_ImportFrom_get_defined_names():
    from parso.python.tree import ImportFrom
    from itertools import islice

    def get_defined_names_o(self: ImportFrom, include_setitem=False):
        last = self.children[-1]

        if last == ")":
            last = self.children[-2]
        elif last == "*":
            return  # No names defined directly.

        if last.type == "import_as_names":
            as_names = islice(last.children, None, None, 2)
        else:
            as_names = [last]

        for name in as_names:
            if name.type != "name":
                name = name.children[-1]
            yield name

    ImportFrom.get_defined_names = get_defined_names_o


# Eliminate use of os.stat and just dump the contents of a stub directory
# into a named tuple. Jedi can deal with whether or not the files are valid.
def optimize_create_stub_map():
    from jedi.inference.gradual.typeshed import _create_stub_map
    from textension.utils import _TupleBase, _named_index
    from itertools import compress
    from operator import methodcaller, not_
    from builtins import list, map, filter
    from os import listdir

    tail = "/__init__.pyi"
    is_ext = methodcaller("__contains__", ".")
    is_pyi = methodcaller("endswith", ".pyi")

    class PathInfo2(_TupleBase):
        path           = _named_index(0)
        is_third_party = _named_index(1)

    def _create_stub_map_p(directory_path_info):
        dirpath = directory_path_info.path

        try:
            listed = listdir(dirpath)
        except (FileNotFoundError, NotADirectoryError):
            return {}

        stubs   = {}
        prefix  = f"{dirpath}/"

        is_third_party = directory_path_info.is_third_party

        has_ext = list(map(is_ext, listed))

        # Directories.
        for directory in compress(listed, map(not_, has_ext)):
            stubs[directory] = PathInfo2((f"{prefix}{directory}{tail}", is_third_party))

        # Single modules.
        for file in filter(is_pyi, compress(listed, has_ext)):
            stubs[file[:-4]] = PathInfo2((f"{prefix}{file}", is_third_party))

        if "__init__" in stubs:
            del stubs["__init__"]

        return stubs

    _patch_function(_create_stub_map, _create_stub_map_p)


# Optimizes _StringComparisonMixin to use a faster check.
def optimize_StringComparisonMixin__eq__():
    from parso.python.tree import _StringComparisonMixin
    from builtins import str

    def __eq__(self, other):
        if other.__class__ is str:
            return self.value == other
        return self is other

    _StringComparisonMixin.__eq__   = __eq__
    _StringComparisonMixin.__hash__ = _forwarder("value.__hash__")


# Optimizes various ValueContext members to use method descriptors.
def optimize_ValueContext():
    from jedi.inference.context import ValueContext

    ValueContext.parent_context = _forwarder("_value.parent_context")
    ValueContext.tree_node      = _forwarder("_value.tree_node")
    ValueContext.name           = _forwarder("_value.name")

    ValueContext.get_value      = _unbound_getter("_value")


def optimize_ClassMixin():
    from jedi.inference.value.klass import ClassMixin, ClassContext

    ClassMixin.is_class       = truthy_noargs
    ClassMixin.is_class_mixin = truthy_noargs

    ClassMixin.py__name__  = _unbound_getter("name.string_name")
    ClassMixin._as_context = _unbound_method(ClassContext)


# Optimizes Compiled**** classes to use forwarding descriptors.
def optimize_Compiled_methods():
    from jedi.inference.compiled.value import CompiledValue, CompiledContext, CompiledModule, CompiledModuleContext

    CompiledValue.get_qualified_names = _forwarder("access_handle.get_qualified_names")

    CompiledValue.is_compiled = truthy_noargs
    CompiledValue.is_stub     = falsy_noargs

    CompiledValue.is_class    = _forwarder("access_handle.is_class")
    CompiledValue.is_function = _forwarder("access_handle.is_function")
    CompiledValue.is_instance = _forwarder("access_handle.is_instance")
    CompiledValue.is_module   = _forwarder("access_handle.is_module")


    CompiledValue.py__bool__  = _forwarder("access_handle.py__bool__")
    CompiledValue.py__doc__   = _forwarder("access_handle.py__doc__")
    CompiledValue.py__name__  = _forwarder("access_handle.py__name__")

    CompiledValue._as_context = _unbound_method(CompiledContext)

    CompiledValue.get_metaclasses = rep_NO_VALUES

    CompiledModule._as_context = _unbound_method(CompiledModuleContext)
    CompiledModule.py__path__  = _forwarder("access_handle.py__path__")
    CompiledModule.py__file__  = _forwarder("access_handle.py__file__")


# Optimizes CompiledName initializer to omit calling parent_value.as_context()
# and instead make it on-demand.
def optimize_CompiledName():
    from jedi.inference.compiled.value import CompiledName, CompiledValue
    from ..tools import state
    from operator import methodcaller
    from textension.utils import _forwarder, _descriptor

    CompiledName._inference_state = state

    def __init__(self: CompiledName, inference_state, parent_value, name):
        self._parent_value = parent_value
        self.string_name = name

    CompiledName.__init__ = __init__

    CompiledValue._context = _descriptor(methodcaller("as_context"))

    CompiledName.parent_context = _forwarder("_parent_value._context")
    CompiledName.get_root_context = _forwarder("parent_context.get_root_context")



# Optimizes TreeContextMixin
# Primarily binds 'infer_node' as a method.
def optimize_TreeContextMixin():
    from jedi.inference.syntax_tree import infer_node
    from jedi.inference.context import TreeContextMixin
    from ..common import state_cache

    TreeContextMixin.infer_node = _unbound_method(state_cache(infer_node))


# Optimizes MergedFilter methods to use functional style calls.
def optimize_MergedFilter():
    from jedi.inference.filters import MergedFilter
    from textension.utils import starchain
    from builtins import list, map
    from operator import methodcaller

    call_values = methodcaller("values")

    @_unbound_method
    def values(self: MergedFilter):
        return list(starchain(map(call_values, self._filters)))

    MergedFilter.values = values


def optimize_shadowed_dict():
    from jedi.inference.compiled.getattr_static import _sentinel
    from jedi.inference.compiled import getattr_static
    from builtins import type
    from types import GetSetDescriptorType

    get_dict = type.__dict__["__dict__"].__get__
    get_mro = type.__dict__["__mro__"].__get__

    def _shadowed_dict(klass):
        for entry in get_mro(klass):
            try:
                class_dict = get_dict(entry)["__dict__"]
            except:  # KeyError
                pass
            else:
                if not (type(class_dict) is GetSetDescriptorType
                   and class_dict.__name__ == "__dict__"
                   and class_dict.__objclass__ is entry):
                       return class_dict
        return _sentinel

    _patch_function(getattr_static._shadowed_dict, _shadowed_dict)


# Eliminates unnecessary checks for known types.
def optimize_getattr_static():
    from jedi.inference.compiled.getattr_static import (
        _sentinel,
        _shadowed_dict,
        _check_instance,
        _safe_hasattr,
        _safe_is_data_descriptor,
        _static_getmro,
        getattr_static as _getattr_static
    )

    from builtins import isinstance, type
    from types import MemberDescriptorType, ModuleType

    common_dicts = {
        type: type.__dict__["__dict__"].__get__,
        object: type.__dict__["__dict__"].__get__,
        ModuleType: ModuleType.__dict__["__dict__"].__get__
    }
    excludes = {ModuleType.__mro__}

    def getattr_static(obj, attr, default=_sentinel):
        instance_result = _sentinel

        if not isinstance(obj, type):
            cls = type(obj)
            if cls in common_dicts:
                mro = cls.__mro__
                try:
                    # Some modules like bpy.types use custom __getattr__.
                    instance_result = common_dicts[cls](obj)[attr]
                except:  # KeyError
                    pass
            else:
                mro = _static_getmro(cls)
                dict_attr = _shadowed_dict(cls)
                if (dict_attr is _sentinel or type(dict_attr) is MemberDescriptorType):
                    instance_result = _check_instance(obj, attr)
        else:
            cls = obj
            mro = _static_getmro(cls)

        cls_result = _sentinel

        # Skip checking known types.
        if mro not in excludes:
            for entry in mro:
                if _shadowed_dict(type(entry)) is _sentinel:
                    try:
                        cls_result = entry.__dict__[attr]
                        break
                    except:  # KeyError
                        pass

        if instance_result is not _sentinel and cls_result is not _sentinel:
            if _safe_hasattr(cls_result, '__get__') \
                    and _safe_is_data_descriptor(cls_result):
                # A get/set descriptor has priority over everything.
                return cls_result, True

        if instance_result is not _sentinel:
            return instance_result, False
        if cls_result is not _sentinel:
            return cls_result, _safe_hasattr(cls_result, '__get__')

        if obj is cls:
            # for types we check the metaclass too
            for entry in _static_getmro(type(cls)):
                if _shadowed_dict(type(entry)) is _sentinel:
                    try:
                        return entry.__dict__[attr], False
                    except:  # KeyError
                        pass
        if default is not _sentinel:
            return default, False
        raise AttributeError(attr)

    _patch_function(_getattr_static, getattr_static)


# Optimizes _static_getmro to read mro.__class__.
def optimize_static_getmro():
    from jedi.inference.compiled import getattr_static
    from jedi.debug import warning
    from builtins import tuple, type, list

    get_mro = type.__dict__['__mro__'].__get__

    def _static_getmro(cls):
        mro = get_mro(cls)
        # Should be safe enough, since the mro itself is obtained via type descriptor.
        if mro.__class__ is tuple or mro.__class__ is list:
            return mro
        warning('mro of %s returned %s, should be a tuple' % (cls, mro))
        return ()

    _patch_function(getattr_static._static_getmro, _static_getmro)


# Optimizes AbstractTreeName to use descriptors for some of its encapsulated properties.
def optimize_AbstractTreeName_properties():
    from jedi.inference.names import AbstractTreeName

    AbstractTreeName.start_pos   = _forwarder("tree_name.start_pos")
    AbstractTreeName.string_name = _forwarder("tree_name.value")


def optimize_BaseName_properties():
    from jedi.inference.names import BaseTreeParamName, AbstractNameDefinition
    from jedi.api.completion import ParamNameWithEquals
    from jedi.api.classes import BaseName

    # Add properties to BaseName subclasses so 'public_name' becomes a descriptor.
    AbstractNameDefinition.get_public_name = _unbound_getter("string_name")
    AbstractNameDefinition.public_name = property(AbstractNameDefinition.get_public_name)
    ParamNameWithEquals.public_name = property(ParamNameWithEquals.get_public_name)
    BaseTreeParamName.public_name = property(BaseTreeParamName.get_public_name)

    BaseName.name = _forwarder("_name.public_name")

    AbstractNameDefinition.get_root_context = _forwarder("parent_context.get_root_context")
    AbstractNameDefinition.api_type = _forwarder("parent_context.api_type")


# Optimizes Leaf to use builtin descriptor for faster access.
def optimize_Leaf():
    from parso.tree import Leaf

    @_forwarder("line", "column").setter
    def start_pos(self: Leaf, pos: tuple[int, int]) -> None:
        self.line, self.column = pos

    Leaf.start_pos = start_pos

    def __init__(self: Leaf, value: str, start_pos: tuple[int, int], prefix: str = ''):
        self.value = value
        self.line, self.column = start_pos
        self.prefix = prefix
        self.parent = None

    Leaf.__init__ = __init__


def optimize_split_lines():
    from parso import split_lines

    def split_lines_o(string: str, keepends: bool = False) -> list[str]:
        lines = string.splitlines(keepends=keepends)
        try:
            if string[-1] is "\n":
                lines += "",
        except:  # Assume IndexError
            lines += "",
        return lines
    _patch_function(split_lines, split_lines_o)


# Optimizes Param.get_defined_names by probability.
def optimize_Param_get_defined_names():
    from parso.python.tree import Param

    def get_defined_names_o(self, include_setitem=False):
        if (ctype := (c := self.children[0]).type) == "name":
            return [c]

        elif ctype == "tfpdef":
            return [c.children[0]]
        
        # Must be ``operator`` at this point.
        elif (c := self.children[1]).type == "tfpdef":
            return [c.children[0]]
        return [c]

    Param.get_defined_names = get_defined_names_o


# Improve the performance of platform.system() for win32.
def optimize_platform_system():
    import platform
    import sys

    if sys.platform == "win32":
        platform.system = "Windows".__str__


def optimize_getmodulename():
    module_suffixes = {suf: -len(suf) for suf in sorted(
        __import__("importlib").machinery.all_suffixes() + [".pyi"], key=len, reverse=True)}
    suffixes = tuple(module_suffixes)
    module_suffixes = module_suffixes.items()
    rpartition = str.rpartition
    endswith = str.endswith
    import inspect

    def getmodulename(name: str):
        if endswith(name, suffixes):
            for suf, neg_suflen in module_suffixes:
                if name[neg_suflen:] == suf:
                    return rpartition(rpartition(name, "\\")[-1], "/")[-1][:neg_suflen]
        return None

    _patch_function(inspect.getmodulename, getmodulename)


# Make is_big_annoying_library into a no-op.
def optimize_is_big_annoying_library():
    from jedi.inference.helpers import is_big_annoying_library

    _patch_function(is_big_annoying_library, lambda _: None)


def optimize_iter_module_names():
    from jedi.inference.compiled.subprocess.functions import _iter_module_names
    from importlib.machinery import all_suffixes
    from os import scandir, DirEntry
    from sys import modules

    from textension.utils import starchain

    module_keys = modules.keys()
    is_dir = DirEntry.is_dir
    endswith = str.endswith
    isidentifier = str.isidentifier
    rsplit = str.rsplit
    suffixes = tuple(set(all_suffixes() + [".pyi"]))

    cache = {}

    # Not identical to the stock function. It allows stupid names like
    # ``__phello__.foo`` which is a frozenlib test import module.
    def _iter_module_names_o(inference_state, paths):
        paths = tuple(paths)
        if paths not in cache:
            names   = []
            entries = []
            _paths  = iter(paths)

            while True:
                try:
                    entries += starchain(map(scandir, _paths))
                    break
                except (FileNotFoundError, NotADirectoryError):
                    pass

            for entry in entries:
                name = entry.name

                if endswith(name, suffixes):
                    name = rsplit(name, ".", 1)[-2]
                    if name != "__init__":
                        names += name,
                elif is_dir(entry) and isidentifier(name) and name != "__pycache__":
                    names += name,

            cache[paths] = set(names)
        return cache[paths]# | set([k for k in module_keys if "." not in k])

    _patch_function(_iter_module_names, _iter_module_names_o)


def optimize_BaseNode_get_last_leaf():
    from parso.tree import BaseNode
    from ..common import node_types

    def get_last_leaf(self: BaseNode) -> tuple[int, int]:
        self = self.children[-1]
        while self.__class__ in node_types:
            self = self.children[-1]
        return self

    BaseNode.get_last_leaf = get_last_leaf


def optimize_Leaf_end_pos():
    from parso.python.tree import Newline
    from parso.tree import Leaf

    strlen = str.__len__

    @property
    def end_pos(self: Leaf) -> tuple[int, int]:
        # The code for newline checks is moved to its own property below.
        return self.line, self.column + strlen(self.value)
    
    Leaf.end_pos = end_pos

    @property
    def end_pos(self: Newline):
        return self.line + 1, 0

    Newline.end_pos = end_pos


# Remove recursion and inline optimized leaf end position. Simplified based on
# ``value`` either being a single newline, or not containing a newline at all.
def optimize_BaseNode_end_pos():
    from parso.tree import BaseNode
    from ..common import node_types
    strlen = str.__len__

    @property
    def end_pos(self: BaseNode) -> tuple[int, int]:
        self = self.children[-1]
        while self.__class__ in node_types:
            self = self.children[-1]

        if self.value is "\n":
            return self.line + 1, 0
        return self.line, self.column + strlen(self.value)

    BaseNode.end_pos = end_pos


# Removes junk indirection code. Jedi uses indirection for multi-process
# inference states. We can't even use that in Blender.
def optimize_CompiledValue_api_type():
    from jedi.inference.compiled.value import CompiledValue
    from types import ModuleType, BuiltinFunctionType, MethodType, \
         FunctionType, MethodDescriptorType

    function_types = MethodDescriptorType, BuiltinFunctionType, MethodType, FunctionType

    is_module = ModuleType.__instancecheck__

    @property
    def api_type(self: CompiledValue):
        obj = self.access_handle.access._obj

        if isinstance(obj, function_types):
            return "function"
        elif isinstance(obj, type):
            return "class"
        elif is_module(obj):
            return "module"
        return "instance"

    CompiledValue.api_type = api_type


def optimize_pickling():
    from parso.cache import _NodeCacheItem
    from parso.pgen2.generator import DFAState, ReservedString, DFAPlan
    from parso.pgen2.grammar_parser import NFAState, NFAArc
    from parso.tree import NodeOrLeaf
    from functools import reduce
    from operator import add

    pool = [NodeOrLeaf, _NodeCacheItem, DFAState, NFAState, NFAArc, ReservedString, DFAPlan]

    for cls in iter(pool):
        cls.__slotnames__ = list(reduce(add, (getattr(c, "__slots__", ()) for c in cls.__mro__)))
        pool += cls.__subclasses__()


# Just removes the context manager. WE DON'T USE PREDEFINED NAMES.
def optimize_LazyTreeValue_infer():
    from jedi.inference.lazy_value import LazyTreeValue

    def infer(self: LazyTreeValue):
        return self.context.infer_node(self.data)
    
    LazyTreeValue.infer = infer


# Just makes name conversions use aggregate initialization.
def optimize_LazyInstanceClassName():
    from jedi.inference.value.klass import ClassName
    from jedi.inference.value.instance import LazyInstanceClassName, InstanceClassFilter
    from textension.utils import _TupleBase, _named_index
    from itertools import repeat
    from builtins import list, map, zip

    class LazyName(_TupleBase, LazyInstanceClassName):
        _instance     = _named_index(0)
        _wrapped_name = _named_index(1)

        tree_name     = _forwarder("_wrapped_name.tree_name")

        def __repr__(self):
            return f"LazyName({object.__repr__(self._wrapped_name)})"

    def _convert(self: InstanceClassFilter, names):
        return list(map(LazyName, zip(repeat(self._instance), names)))

    InstanceClassFilter._convert = _convert


# Optimize get_module_info to use already-imported modules if they exist, in
# order to avoid invoking the import machinery which isn't actually cheap.
def optimize_get_module_info():
    from jedi.inference.compiled.subprocess.functions import get_module_info, _find_module
    from jedi.file_io import KnownContentFileIO
    from types import ModuleType
    import sys

    modules = sys.modules
    module_getattr = ModuleType.__getattribute__

    def _get_module_info(inference_state, sys_path=None, full_name=None, **kwargs):
        if full_name in modules:
            try:
                spec = module_getattr(modules[full_name], "__spec__")
                with open(spec.origin, "rb") as f:
                    content = f.read()
                return KnownContentFileIO(spec.origin, content), bool(spec.submodule_search_locations)
            except:
                pass

        temp = sys.path
        if sys_path is not None:
            sys.path = sys_path

        try:
            return _find_module(full_name=full_name, **kwargs)
        except ImportError:
            return None, None
        finally:
            if sys_path is not None:
                sys.path = temp

    _patch_function(get_module_info, _get_module_info)


def optimize_tree_name_to_values():
    from jedi.inference.syntax_tree import tree_name_to_values
    from ..common import state_cache

    from jedi.inference.syntax_tree import ValueSet, infer_atom, ContextualizedNode, iterate_values, TreeNameDefinition, check_tuple_assignments, infer_expr_stmt, _apply_decorators, infer_node
    from jedi.inference.gradual import annotation
    from jedi.inference import imports

    def infer_with(context, node, tree_name):
        if types := annotation.find_type_from_comment_hint_with(context, node, tree_name):
            return types
        value_managers = context.infer_node(node.get_test_node_from_name(tree_name))

        if node.parent.type == 'async_stmt':
            enter_methods = value_managers.py__getattribute__('__aenter__')
            coro = enter_methods.execute_with_values()
            return coro.py__await__().py__stop_iteration_returns()

        enter_methods = value_managers.py__getattribute__('__enter__')
        return enter_methods.execute_with_values()

    def infer_for(context, node, tree_name):
        if node.type == 'for_stmt' and (types := annotation.find_type_from_comment_hint_for(context, node, tree_name)):
            return types

        # XXX: Removed predefined names code. Jedi doesn't even use it.
        cn = ContextualizedNode(context, node.children[3])
        is_async = node.parent.type == 'async_stmt'
        for_types = iterate_values(cn.infer(), contextualized_node=cn, is_async=is_async)
        n = TreeNameDefinition(context, tree_name)
        return check_tuple_assignments(n, for_types)

    # This version searches for annotations more efficiently.
    @state_cache
    def _tree_name_to_values(inference_state, context, tree_name):
        n = tree_name
        while n := n.parent:
            if n.type == "expr_stmt":
                if n.children[1].type == "annassign":
                    return annotation.infer_annotation(context, n.children[1].children[1]).execute_annotation()
                break

        node = tree_name.get_definition(import_name_always=True, include_setitem=True)
        if node is None:
            node = tree_name.parent
            if node.type == 'global_stmt':
                c = context.create_context(tree_name)
                if c.is_module():
                    return NO_VALUES
                names = next(c.get_filters()).get(tree_name.value)
                return ValueSet.from_sets(name.infer() for name in names)
            elif node.type not in ('import_from', 'import_name'):
                return infer_atom(context.create_context(tree_name), tree_name)

        typ = node.type

        if typ in {"with_stmt", "for_stmt", "comp_for", "sync_comp_for"}:
            if typ == 'with_stmt':
                return infer_with(context, node, tree_name)
            return infer_for(context, node, tree_name)

        elif typ == 'expr_stmt':
            return infer_expr_stmt(context, node, tree_name)
        elif typ in ('import_from', 'import_name'):
            return imports.infer_import(context, tree_name)
        elif typ in ('funcdef', 'classdef'):
            return _apply_decorators(context, node)
        elif typ == 'try_stmt':
            exceptions = context.infer_node(tree_name.get_previous_sibling().get_previous_sibling())
            return exceptions.execute_with_values()
        elif typ == 'param':
            return NO_VALUES
        elif typ == 'del_stmt':
            return NO_VALUES
        elif typ == 'namedexpr_test':
            return infer_node(context, node)
        else:
            raise ValueError("Should not happen. type: %s" % typ)

    _patch_function(tree_name_to_values, _tree_name_to_values)


def optimize_infer_expr_stmt():
    from jedi.inference.syntax_tree import infer_expr_stmt, _infer_expr_stmt

    # Skip recursion safe wrappers.
    _patch_function(infer_expr_stmt, _infer_expr_stmt.__closure__[0].cell_contents)


def optimize_ClassMixin_py__mro__():
    from jedi.inference.value.klass import ClassMixin

    # Skip recursion safe wrappers.
    ClassMixin.py__mro__ = ClassMixin.py__mro__.__closure__[0].cell_contents


def optimize_Param_name():
    from parso.python.tree import Param, Operator

    @property
    def name(self: Param):
        name = self.children[0]
        if name.type == "name":
            return name

        # XXX: Can we just check ``name.value`` here? Are there other operator types for param at position 0?
        # Could be star unpack.
        elif name.type == "operator" and name.value in "**":
            name = self.children[1]
        # Could also be nested after operator.
        if name.type == "tfpdef":
            return name.children[0]
        return name
    
    Param.name = name


def optimize_CompiledInstanceName():
    from jedi.inference.value.instance import CompiledInstanceName, CompiledInstanceClassFilter
    from textension.utils import _TupleBase, _named_index
    from builtins import list, map, zip

    class InstanceName(_TupleBase, CompiledInstanceName):
        _wrapped_name = _named_index(0)

        # This just makes optimized filter_names faster.
        tree_name     = None

        def __repr__(self):
            return '%s(%s)' % (self.__class__.__name__, self._wrapped_name)

    def _convert(self, names):
        return list(map(InstanceName, zip(names)))

    CompiledInstanceClassFilter._convert = _convert


def optimize_AbstractContext_get_root_context():
    from jedi.inference.context import AbstractContext

    def get_root_context(self):
        while parent_context := self.parent_context:
            self = parent_context
        return self

    AbstractContext.get_root_context = get_root_context


def optimize_NodesTree_copy_nodes():
    from parso.python.diff import _NodesTree, _NodesTreeNode, _func_or_class_has_suite, _is_flow_node, _ends_with_newline, _get_suite_indentation, split_lines, _get_indentation
    from itertools import islice
    from ..common import node_types

    def _copy_nodes(self, working_stack, nodes, until_line, line_offset, prefix='', is_nested=False):
        new_nodes = []
        added_indents = []

        if is_nested:
            indent = nodes[1].start_pos[1]
            nodes = islice(nodes, 1, None)
        else:
            indent = nodes[0].start_pos[1]


        if is_nested or indent in self.indents:
            for node in nodes:
                if node.type in {"endmarker", "error_leaf"}:
                    if node.type != "error_leaf" or node.token_type in {"DEDENT", "ERROR_DEDENT"}:
                        break

                # Get the node's first leaf.
                first_leaf = node
                while first_leaf.__class__ in node_types:
                    first_leaf = first_leaf.children[0]

                if first_leaf.column == indent:
                    if first_leaf.line > until_line:
                        break

                    # Get the last leaf.
                    last_leaf = node
                    while last_leaf.__class__ in node_types:
                        last_leaf = last_leaf.children[-1]

                    # The last line, unless the last leaf is a newline.
                    last_line = last_leaf.line

                    if last_leaf.type != "newline":

                        # Get the next leaf.
                        next_leaf = node
                        while next_leaf is (children := next_leaf.parent.children)[-1]:
                            if next_leaf := next_leaf.parent:
                                continue
                            next_leaf = None
                            break
                        i = -1
                        while next_leaf is not children[i]:
                            i -= 1
                        next_leaf = children[i + 1]
                        while next_leaf.__class__ in node_types:
                            next_leaf = next_leaf.children[0]

                        if next_leaf.type == "endmarker" and "\n" in next_leaf.prefix:
                            last_line = last_leaf.line + 1

                    if last_line > until_line:
                        if _func_or_class_has_suite(node):
                            new_nodes += node,
                        break

                    # with any.measure_total:
                    if node.__class__ in node_types:

                        if node.type in {"decorated", "async_funcdef", "async_stmt"}:
                            n = node
                            if n.type == "decorated":
                                n = n.children[-1]
                            if n.type in {"async_funcdef", "async_stmt"}:
                                n = n.children[-1]
                            if n.type in {"classdef", "funcdef"}:
                                suite_node = n.children[-1]
                            else:
                                suite_node = node.children[-1]
                        else:
                            suite_node = node.children[-1]

                        if suite_node.type in {'error_leaf', 'error_node'}:
                            break

                    new_nodes += node,

        # Pop error nodes at the end from the list
        if new_nodes:
            while new_nodes:
                last_node = new_nodes[-1]
                if (last_node.type in ('error_leaf', 'error_node')
                        or _is_flow_node(new_nodes[-1])):
                    new_nodes.pop()
                    while new_nodes:
                        last_node = new_nodes[-1]
                        if last_node.get_last_leaf().type == 'newline':
                            break
                        new_nodes.pop()
                    continue
                if len(new_nodes) > 1 and new_nodes[-2].type == 'error_node':
                    new_nodes.pop()
                    continue
                break

        if not new_nodes:
            return [], working_stack, prefix, added_indents

        new_prefix = ''
        tos = working_stack[-1]
        last_node = new_nodes[-1]
        had_valid_suite_last = False

        # Pop incomplete suites from the list
        if _func_or_class_has_suite(last_node):
            suite = last_node
            while suite.type != 'suite':
                suite = suite.children[-1]

            indent = _get_suite_indentation(suite)
            added_indents += indent,

            suite_tos = _NodesTreeNode(suite, indentation=_get_indentation(last_node))
            suite_nodes, new_working_stack, new_prefix, ai = self._copy_nodes(
                working_stack + [suite_tos], suite.children, until_line, line_offset,
                is_nested=True,
            )
            added_indents += ai
            if len(suite_nodes) < 2:
                new_nodes.pop()
                new_prefix = ''
            else:
                assert new_nodes
                tos.add_child_node(suite_tos)
                working_stack = new_working_stack
                had_valid_suite_last = True

        if new_nodes:
            if not _ends_with_newline(new_nodes[-1].get_last_leaf()) and not had_valid_suite_last:
                p = new_nodes[-1].get_next_leaf().prefix
                new_prefix = split_lines(p, keepends=True)[0]

            if had_valid_suite_last:
                last = new_nodes[-1]
                if last.type == 'decorated':
                    last = last.children[-1]
                if last.type in ('async_funcdef', 'async_stmt'):
                    last = last.children[-1]
                last_line_offset_leaf = last.children[-2].get_last_leaf()
                assert last_line_offset_leaf == ':'
            else:
                last_line_offset_leaf = new_nodes[-1].get_last_leaf()
            tos.add_tree_nodes(
                prefix, new_nodes, line_offset, last_line_offset_leaf,
            )
            prefix = new_prefix
            self._prefix_remainder = ''

        return new_nodes, working_stack, prefix, added_indents

    _NodesTree._copy_nodes = _copy_nodes


def optimize_NodeOrLeaf_get_next_leaf():
    from parso.tree import NodeOrLeaf
    from ..common import node_types

    def get_next_leaf(self: NodeOrLeaf):
        if self.parent is None:
            return None

        while self is (children := self.parent.children)[-1]:
            if self := self.parent:
                continue
            return None

        i = -1
        while self is not children[i]:
            i -= 1

        self = children[i + 1]
        while self.__class__ in node_types:
            self = self.children[0]
        return self

    NodeOrLeaf.get_next_leaf = get_next_leaf


def optimize_find_overload_functions():
    from jedi.inference.value.function import _find_overload_functions
    from itertools import compress, repeat
    from operator import attrgetter, eq
    from builtins import map

    get_type = attrgetter("type")
    rep_decorated = repeat("decorated")

    def find_overload_functions(context, node):
        ret = []

        if node.type == "funcdef":
            scope = node.parent
            name_str = node.children[1].value
            if scope.type == "decorated":
                scope = scope.parent

            children  = scope.children
            selectors = map(eq, rep_decorated, map(get_type, children))

            for dec in compress(children, selectors):
                funcdef = dec.children[1]
                if funcdef.children[1].value == name_str:

                    # Can be ``decorator`` or ``decorators`` (plural).
                    # The latter contains a list of ``decorator`` nodes.
                    dec = dec.children[0]
                    if dec.type == "decorators":
                        dec = dec.children[0]

                    if dec.children[1].value == "overload":
                        ret += funcdef,
        return ret

    _patch_function(_find_overload_functions, find_overload_functions)


def optimize_builtin_from_name():
    from jedi.inference.gradual.typeshed import _load_from_typeshed
    from jedi.inference.compiled.value import create_cached_compiled_value
    from jedi.inference.compiled import builtin_from_name
    from jedi.inference import InferenceState, ValueSet
    from ..tools import get_handle, state
    import builtins

    @property
    def builtins_module(self):
        handle = get_handle(builtins)
        value  = create_cached_compiled_value(state, handle, None)
        py_value = ValueSet((value,))

        stub_value = _load_from_typeshed(state, py_value, None, ("builtins",))

        # Mutate property so builtins is loaded only once.
        InferenceState.builtins_module = stub_value
        return stub_value

    InferenceState.builtins_module = builtins_module

    cache = {}

    def _builtin_from_name(inference_state, string):
        if string not in cache:
            stub_value = inference_state.builtins_module
            if string in ('None', 'True', 'False'):
                builtins, = stub_value.non_stub_value_set
                filter_ = next(builtins.get_filters())
            else:
                filter_ = next(stub_value.get_filters())
            name, = filter_.get(string)
            value, = name.infer()
            cache[string] = value
        return cache[string]
    
    _patch_function(builtin_from_name, _builtin_from_name)