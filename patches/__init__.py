# This adds fixes for jedi to outright work.

from textension.utils import _patch_function, _forwarder
from .tools import _descriptor_overrides, _value_overrides

import bpy


def apply():
    # Needs to run before any optimization that modifies pickled objects.
    patch_load_from_file_system()

    _apply_optimizations()
    _apply_patches()


def _apply_patches():
    from . import modules
    modules._bpy_types.apply()
    modules._mathutils.apply()
    modules._gpu.apply()
    modules._bpy.apply()

    patch_NameWrapper_getattr()
    patch_Completion_complete_inherited()
    patch_Value_py__getattribute__alternatives()
    patch_fakelist_array_type()
    patch_import_resolutions()
    patch_get_builtin_module_names()
    patch_compiledvalue()  # XXX: Not having this means tuple[X] isn't subscriptable.
    # patch_misc()  # XXX: This is very experimental.
    patch_is_pytest_func()
    patch_paths_from_list_modifications()

    patch_create_cached_compiled_value()
    patch_various_redirects()
    patch_SequenceLiteralValue()
    patch_complete_dict()
    patch_convert_values()
    patch_get_user_context()
    patch_Importer()
    patch_get_importer_names()


def _apply_optimizations():
    from . import opgroup
    opgroup.interpreter.apply()
    opgroup.safe_optimizations.apply()
    opgroup.defined_names.apply()
    opgroup.completions.apply()
    opgroup.used_names.apply()
    opgroup.filters.apply()
    opgroup.memo.apply()
    opgroup.py_getattr.apply()


# Jedi doesn't consider the indentation at the completion site potentially
# giving the wrong scope at the last line of the scope. This fixes that.
def patch_get_user_context():
    from jedi.inference.context import ModuleContext
    from jedi.api import completion

    def get_user_context(module_context: ModuleContext, pos: tuple[int]):
        leaf = module_context.tree_node.get_leaf_for_position(pos, include_prefixes=True)

        if leaf.type == "newline":
            if leaf is leaf.parent.get_last_leaf():
                if pos[1] < leaf.parent.start_pos[1]:
                    leaf = leaf.get_next_leaf()

        elif leaf.type == "endmarker" or leaf.start_pos > pos:
            if last := leaf.get_previous_leaf():
                 if last.type == "newline":
                    if pos[1] == last.parent.start_pos[1]:
                        leaf = last

        return module_context.create_context(leaf)

    _patch_function(completion.get_user_context, get_user_context)


# This fixes jedi attempting to complete from both stubs and compiled modules,
# which isn't allowed by PEP 484. It's either or.
def patch_convert_values():
    from jedi.inference.gradual import conversion

    # This makes ``prefer_stubs`` True by default and False if ``only_stubs`` is True.
    def convert_values(values, only_stubs=False, prefer_stubs=True, ignore_compiled=True):
        if only_stubs:
            prefer_stubs = False
        return convert_values(values, only_stubs=only_stubs, prefer_stubs=prefer_stubs, ignore_compiled=ignore_compiled)
    convert_values = _patch_function(conversion.convert_values, convert_values)


# Fix unpickling errors jedi doesn't catch.
def patch_load_from_file_system():
    from parso.cache import _load_from_file_system

    def safe_wrapper(*args, **kw):
        try:
            return _load_from_file_system(*args, **kw)
        except ModuleNotFoundError:
            return None  # Return None so jedi can re-save it.
        except AttributeError:
            return None
        
    _load_from_file_system = _patch_function(_load_from_file_system, safe_wrapper)


def patch_NameWrapper_getattr():
    from jedi.inference.names import NameWrapper

    NameWrapper.public_name     = _forwarder("_wrapped_name.public_name")
    NameWrapper.get_public_name = _forwarder("_wrapped_name.get_public_name")
    NameWrapper.string_name     = _forwarder("_wrapped_name.string_name")
    NameWrapper.parent_context  = _forwarder("_wrapped_name.parent_context")


def patch_SequenceLiteralValue():
    from jedi.inference.gradual.base import GenericClass, _LazyGenericBaseClass
    from jedi.cache import memoize_method

    @memoize_method
    def py__bases__(self: GenericClass):
        ret = []
        for base in self._wrapped_value.py__bases__():
            ret += [_LazyGenericBaseClass(self, base, self._generics_manager)]
        return ret

    GenericClass.py__bases__ = py__bases__

    from jedi.inference.gradual.typing import TypedDict
    from jedi.inference.base_value import ValueSet
    from jedi.inference.value import TreeInstance

    @memoize_method
    def py__call__(self: GenericClass, arguments):
        if self.is_typeddict():
            return ValueSet([TypedDict(self)])
        return ValueSet([TreeInstance(self.inference_state, self.parent_context, self, arguments)])

    GenericClass.py__call__ = py__call__


# Removes calls to __getattr__ for dynamic forwarding. This is an effort to
# eliminate most stack overflows during debugging.
def patch_various_redirects():
    from jedi.inference.value.iterable import SequenceLiteralValue

    SequenceLiteralValue.parent_context = _forwarder("_wrapped_value.parent_context")
    SequenceLiteralValue.py__class__    = _forwarder("_wrapped_value.py__class__")
    SequenceLiteralValue._arguments     = _forwarder("_wrapped_value._arguments")
    SequenceLiteralValue.is_module      = _forwarder("_wrapped_value.is_module")
    SequenceLiteralValue.is_stub        = _forwarder("_wrapped_value.is_stub")
    SequenceLiteralValue.is_instance    = _forwarder("_wrapped_value.is_instance")

    from jedi.inference.gradual.base import GenericClass

    GenericClass.parent_context      = _forwarder("_wrapped_value.parent_context")
    GenericClass.inference_state     = _forwarder("_wrapped_value.inference_state")
    GenericClass.get_metaclasses     = _forwarder("_wrapped_value.get_metaclasses")
    GenericClass.is_bound_method     = _forwarder("_wrapped_value.is_bound_method")
    GenericClass.get_qualified_names = _forwarder("_wrapped_value.get_qualified_names")
    GenericClass.tree_node           = _forwarder("_wrapped_value.tree_node")
    GenericClass.is_compiled         = _forwarder("_wrapped_value.is_compiled")
    GenericClass.list_type_vars      = _forwarder("_wrapped_value.list_type_vars")
    GenericClass.is_stub             = _forwarder("_wrapped_value.is_stub")
    GenericClass.is_instance         = _forwarder("_wrapped_value.is_instance")
    
    from jedi.inference.gradual.type_var import TypeVar, TypeVarClass

    TypeVar.get_safe_value       = _forwarder("_wrapped_value.get_safe_value")
    TypeVar.is_compiled          = _forwarder("_wrapped_value.is_compiled")
    TypeVarClass.inference_state = _forwarder("_wrapped_value.inference_state")
    TypeVarClass.parent_context  = _forwarder("_wrapped_value.parent_context")
    TypeVarClass.is_bound_method = _forwarder("_wrapped_value.is_bound_method")
    TypeVarClass.is_instance     = _forwarder("_wrapped_value.is_instance")
    TypeVarClass.tree_node       = _forwarder("_wrapped_value.tree_node")

    from jedi.inference.gradual.stub_value import VersionInfo

    VersionInfo.get_safe_value = _forwarder("_wrapped_value.get_safe_value")
    VersionInfo.is_compiled    = _forwarder("_wrapped_value.is_compiled")

    from jedi.inference.gradual.typing import TypingClassWithGenerics

    TypingClassWithGenerics.is_compiled = _forwarder("_wrapped_value.is_compiled")
    TypingClassWithGenerics.tree_node   = _forwarder("_wrapped_value.tree_node")
    TypingClassWithGenerics.is_stub     = _forwarder("_wrapped_value.is_stub")
    TypingClassWithGenerics.is_instance = _forwarder("_wrapped_value.is_instance")

    from jedi.inference.compiled import ExactValue

    ExactValue.py__class__ = _forwarder("_wrapped_value.py__class__")


# Fixes jedi erroneously yielding wrong inherited methods.
def patch_Completion_complete_inherited():
    from jedi.api.completion import Completion
    from parso.python.tree import search_ancestor

    def _complete_inherited(self: Completion, is_function=True):
        leaf = self._module_node.get_leaf_for_position(self._position, include_prefixes=True)
        cls = search_ancestor(leaf, 'classdef')
        if cls is None:
            return

        # Complete the methods that are defined in the super classes.
        class_value = self._module_context.create_value(cls)

        if cls.start_pos[1] >= leaf.start_pos[1]:
            return

        filters = class_value.get_filters(is_instance=True)
        # The first dict is the dictionary of class itself.
        next(filters)
        for filter in filters:
            for name in filter.values():
                # TODO we should probably check here for properties
                ret = name.api_type

                # XXX: Here's the fix.
                if ret == 'function' and is_function:
                    yield name

    Completion._complete_inherited = _complete_inherited


# Fallback for py__getattribute__.
def patch_Value_py__getattribute__alternatives():
    from jedi.inference.compiled.value import CompiledValue, Value, NO_VALUES
    from jedi.inference.value.instance import CompiledInstance, ValueSet

    from .tools import make_compiled_value

    # TODO: Make py__getattribute__alternatives for each (CompiledValue, CompiledInstance)
    #       if this patch is still applicable.
    def py__getattribute__alternatives(self: Value, name_or_str):
        if isinstance(self, CompiledInstance):
            value = self.class_value
            if desc_map := _descriptor_overrides.get(self.class_value.access_handle.access._obj):
                if ret := desc_map.get(name_or_str):
                    return make_compiled_value(ret, value.as_context()).py__call__(None)
                else:
                    print("missing descriptor", self, name_or_str)

        elif isinstance(self, CompiledValue):
            obj = self.access_handle.access._obj
            try:
                if obj in _descriptor_overrides:
                    descriptor = getattr(obj, name_or_str)
                    return ValueSet((make_compiled_value(descriptor, self.as_context()),))
            except TypeError:
                pass

        # print("no fallbacks found for", repr(name_or_str))
        return NO_VALUES
    
    Value.py__getattribute__alternatives = py__getattribute__alternatives


# Jedi breaks if there's ill-formed code related to sys path modifications.
# Think ``sys.path.append(foo, bar)``, forgetting a set of parentheses.
# Obviously such code won't run, but it renders Jedi unusable as it raises
# an exception and prevents further completions in the source from working.
def patch_paths_from_list_modifications():
    from jedi.inference.sys_path import _paths_from_list_modifications

    def _safe_wrapper(*args, **kw):
        try:
            # Materialize so this generator can blow up at the call site.
            return list(orig_fn(*args, **kw))

        except BaseException:  # We don't care about the exception.
            return ()
    orig_fn = _patch_function(_paths_from_list_modifications, _safe_wrapper)


# Patch ``_is_pytest_func`` to return False. Otherwise we can't have a
# compiled called ``test`` without jedi thinking it's somehow pytest related.
def patch_is_pytest_func():
    from jedi.plugins.pytest import _is_pytest_func
    
    _patch_function(_is_pytest_func, lambda *_: False)


# Patches Jedi's import resolutions to allow completing sub-modules assigned
# either at runtime or from built-in, file-less modules. This is required for
# import completions of both compiled and mixed file/runtime modules like bpy,
# gpu, mathutils, etc. Names are obtained from compiled modules using safety
# principles similar to how Jedi does things in getattr_static.
def patch_import_resolutions():
    from jedi.inference.compiled.value import CompiledModule, ValueSet
    from jedi.inference.imports import import_module

    from .tools import get_handle, state
    import importlib

    module_cache = state.module_cache

    # Required for builtin module import fallback when Jedi fails.
    def import_module_override(state, name, parent_module, sys_path, prefer_stubs):
        ret = import_module(state, name, parent_module, sys_path, prefer_stubs)
        if not ret:
            try:
                module = importlib.import_module(".".join(name))
                access_handle = get_handle(module)
                ret = ValueSet((CompiledModule(state, access_handle),))
                module_cache.add(name, ret)
            except ImportError:
                pass
        return ret

    import_module = _patch_function(import_module, import_module_override)


# Patches Importer to fix several problems.
# - Make ``_str_import_path`` more efficient to compute
# - Remove workarounds for useless frameworks (flask)
# - Support completing submodules from compiled modules
# - Support completing non-modules (typing.io)
# - Support intercepting imports to provide virtual modules.
#
# Virtual modules are needed because:
# 1. bpy/__init__.py is inferred by source text, not its runtime members.
# 2. bpy.app is not a module and relies on a loophole in import resolution.
# 3. _bpy has no spec/loader causing module detection heuristics to fail.
# 4. bpy.ops uses getattr and dynamic modules implying code execution.
# All which are reasons for Jedi completing them partially or not at all.
# So instead of addressing each problem differently, we use virtual modules.
def patch_Importer():
    from jedi.inference.imports import Importer, ValueSet
    from .common import Importer_redirects, state, is_namenode

    Importer._inference_state = state
    Importer._fixed_sys_path = None
    Importer._infer_possible = True

    # Remove the property. We write the value just once in __init__.
    Importer._str_import_path = None

    # Import redirection for actual modules.
    def follow(self: Importer):
        if module := Importer_redirects.get(".".join(self._str_import_path)):
            return ValueSet((module,))
        return follow(self)

    follow = _patch_function(Importer.follow, follow)

    # Import redirection for submodule names.
    def completion_names(self: Importer, inference_state, only_modules=False):
        if module := Importer_redirects.get(".".join(self._str_import_path)):
            return module.get_submodule_names(only_modules=only_modules)
        return completion_names(self, state, only_modules=only_modules)

    completion_names = _patch_function(Importer.completion_names, completion_names)

    def __init__(self: Importer, inference_state, import_path, module_context, level=0):
        # Defer to original. Don't want to pull that mess here.
        if level > 0:
            __init__(self, state, import_path, module_context, level)
        else:
            self.level = level
            self._module_context = module_context
            # Can be strings, Names or even a mix of both, which is annoying.
            self.import_path = import_path

        # If initialization was deferred to the original, convert here.
        self.import_path = self._str_import_path = tuple(s.value if is_namenode(s) else s for s in self.import_path)

    __init__ = _patch_function(Importer.__init__, __init__)

    # The ONLY place jedi passes a list of NAMES to the importer, and for some
    # reason thought it would be a good idea to keep it like that (it's not).
    # This also removes the try/except clause for a more performant version.
    from jedi.inference.imports import _prepare_infer_import, search_ancestor
    from parso.python.tree import ImportFrom

    is_import_from = ImportFrom.__instancecheck__

    def prepare_infer_import(module_context, tree_name):
        import_node = search_ancestor(tree_name, 'import_name', 'import_from')
        import_path = import_node.get_path_for_name(tree_name)
        from_name = None

        if is_import_from(import_node):
            from_names = import_node.get_from_names()
            if len(from_names) + 1 == len(import_path):
                from_name = import_path[-1]
                import_path = from_names

        import_path = tuple(n.value for n in import_path)
        importer = Importer(state, import_path, module_context, import_node.level)
        return from_name, import_path, import_node.level, importer.follow()

    _patch_function(_prepare_infer_import, prepare_infer_import)


# Fixes completion for "import _bpy" and other file-less modules which
# are not listed in sys.builtin_module_names.
def patch_get_builtin_module_names():
    from jedi.inference.compiled.subprocess import functions
    from builtins import set
    import sys

    builtin = {*sys.builtin_module_names}
    module_items = sys.modules.items()
    get = object.__getattribute__

    def get_builtin_module_names(_):
        tmp = []
        for name, module in module_items:
            if "." in name:
                continue

            try:
                if get(module, "__file__"):
                    continue
            # TypeError or AttributeError.
            except:
                tmp += [name]
        return builtin | set(tmp)
    
    _patch_function(functions.get_builtin_module_names, get_builtin_module_names)


def patch_misc():
    from importlib import _bootstrap_external
    from importlib._bootstrap_external import path_sep, path_separators
    from itertools import repeat
    from builtins import filter
    import os
    import bpy

    def restore():
        _bootstrap_external._path_stat = org_path_stat
        cache.clear()

    def init(path):
        _bootstrap_external._path_stat = fast_path_stat
        bpy.app.timers.register(restore)
        return fast_path_stat(path)

    class Cache(dict):
        def __missing__(self, path):
            ret = self[path] = stat(path)
            return ret

    cache = Cache()
    stat = os.stat
    org_path_stat = _bootstrap_external._path_stat
    fast_path_stat = cache.__getitem__
    _bootstrap_external._path_stat = init

    path_seps = repeat(path_separators)
    join = path_sep.join
    rstrip = str.rstrip

    def _path_join(*parts):
        return join(map(rstrip, filter(None, parts), path_seps))
    _patch_function(_bootstrap_external._path_join, _path_join)


# Patches FakeList to have its 'array_type' class attribute fixed.
def patch_fakelist_array_type():
    from jedi.inference.value.iterable import FakeList

    FakeList.array_type = "list"


# Patches CompiledValue to support inference and extended py__call.
# Specifically fixes [].copy().
#                              ^
def patch_compiledvalue():
    from jedi.inference.compiled.access import create_access_path
    from jedi.inference.compiled.value import CompiledValue, create_from_access_path

    import typing

    AliasTypes = (typing.GenericAlias, typing._GenericAlias)

    # Try to convert GenericAlias into _GenericAlias, the old type.
    def convert_alias(alias):
        if not isinstance(alias, AliasTypes):
            return alias
        args = tuple(convert_alias(a) for a in getattr(alias, "__args__"))
        origin = getattr(typing, alias.__origin__.__name__.capitalize(), alias)
        return typing._GenericAlias(origin, args)

    def py__call__(self: CompiledValue, arguments):
        obj = self.access_handle.access._obj

        if isinstance(obj, typing.GenericAlias):
            a = create_access_path(self.inference_state, convert_alias(obj))
            return create_from_access_path(self.inference_state, a).execute_annotation()
        return py__call__(self, arguments)

    py__call__ = _patch_function(CompiledValue.py__call__, py__call__)


# Patches handle-to-CompiledValue creation to intercept compiled objects.
# Needed as part of RNA inference integration.
def patch_create_cached_compiled_value():
    from jedi.inference.compiled.value import (
        CompiledModule, CompiledValue, create_cached_compiled_value)

    # StructMetaPropGroup is used by bpy_types.Operator, as non-RNA base.
    from bpy_types import RNAMeta, StructRNA, StructMetaPropGroup
    from .modules._bpy_types import get_rna_value

    is_compiled_value = CompiledValue.__instancecheck__
    bpy_types = (RNAMeta, StructRNA, StructMetaPropGroup, bpy.props._PropertyDeferred)
    from .common import CompiledModule_redirects, state, state_cache

    @state_cache
    def create_cached_compiled_value_p(state, handle, context):
        obj = handle.access._obj

        if context:
            assert not is_compiled_value(context)
            if isinstance(obj, bpy_types):
                return get_rna_value(obj, context._value)
            value = CompiledValue(state, handle, context)

        else:
            value = None
            try:
                if obj in CompiledModule_redirects:
                    value = CompiledModule_redirects[obj]
            except TypeError:
                pass

            if value is None:
                value = CompiledModule(state, handle, context)

        # Some objects are unhashable, but they are too rare and not
        # worth the extra overhead of using isinstance for.
        try:
            if obj in _value_overrides:
                _value_overrides[obj](obj, value)
        except TypeError:
            pass

        return value

    _patch_function(create_cached_compiled_value, create_cached_compiled_value_p)

    from jedi.inference.compiled import value
    # The only thing these functions below do different from the stock ones,
    # is that they don't pass the ``parent_context`` as a keyword argument.
    # Wwhy jedi calls ``create_cached_compiled_value`` with keyword arguments,
    # only to convert them back to positional with a wrapper is beyond me.

    def create_from_name(_, compiled_value, name):
        access_paths = compiled_value.access_handle.getattr_paths(name, default=None)

        value = None
        for access_path in access_paths:
            value = create_cached_compiled_value(state, access_path, value and value.as_context())
        return value

    def create_from_access_path(_, access_path):
        value = None
        for name, access in access_path.accesses:
            value = create_cached_compiled_value(state, access, value and value.as_context())
        return value
    
    _patch_function(value.create_from_name, create_from_name)
    _patch_function(value.create_from_access_path, create_from_access_path)


# Patch strings.complete_dict because of a possible AttributeError.
def patch_complete_dict():
    from jedi.api import strings

    def complete_dict(*args, **kw):
        try:
            return complete_dict(*args, **kw)
        # ``before_bracket_leaf`` can be None, but Jedi doesn't check this.
        except AttributeError:
            return []

    complete_dict = _patch_function(strings.complete_dict, complete_dict)


# Fixes jedi not completing fake submodules (bpy.app.x, typing.io, typing.re)
# because they are non-modules with an actual dot "." in their module name.
# Even the standard library does this, so I guess we're fixing this, too?
def patch_get_importer_names():
    from jedi.inference.imports import ImportName
    from jedi.api.completion import Completion, _gather_nodes
    from itertools import repeat, compress
    import sys
    from .tools import is_namenode, is_operator

    startswith = str.startswith
    modules = sys.modules.keys()

    def get_importer_names(self: Completion, names, level=0, only_modules=True):
        # This fix applies only when there's at least 1 import name and 1 dot
        # operator. In this context, dots, comma and parentheses may appear.
        for op in names and filter(is_operator, _gather_nodes(self.stack)):
            if op.value is ".":
                # Compose the import statement and look for it in sys.modules.
                comp = ".".join(n.value for n in names)

                # Jedi may omit the trailing part from names. Add it back.
                leaf = self._module_node.get_leaf_for_position(self._original_position)
                if leaf not in names:
                    if is_namenode(leaf):
                        comp += "."
                    comp += leaf.value

                ret = []
                prefix = comp[:comp.rindex(".") + 1]
                for name in compress(modules, map(startswith, modules, repeat(comp))):
                    # Remove the prefix up to and including the leading dot.
                    name = name.removeprefix(prefix)

                    # If we're completing ``bpy.a``, we only want ``app`` and not
                    # ``app.handlers``.
                    if "." not in name:
                        ret += [name]
                if ret:
                    return [ImportName(self._module_context, n) for n in ret]

        return get_importer_names(self, names, level=level, only_modules=only_modules)

    get_importer_names = _patch_function(Completion._get_importer_names, get_importer_names)
