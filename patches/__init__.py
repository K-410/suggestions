# This adds fixes for jedi to outright work.

from textension.utils import _patch_function, _forwarder
from .tools import state, _descriptor_overrides, _value_overrides, \
    _get_unbound_super_method

import bpy


def apply():
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
    # patch_Importer_follow()
    patch_get_builtin_module_names()
    patch_compiledvalue()  # XXX: Not having this means tuple[X] isn't subscriptable.
    # patch_misc()  # XXX: This is very experimental.
    patch_is_pytest_func()
    patch_paths_from_list_modifications()

    patch_create_cached_compiled_value()
    patch_various_redirects()
    patch_SequenceLiteralValue()
    patch_load_from_file_system()
    patch_complete_dict()
    patch_convert_values()
    patch_get_user_context()


def _apply_optimizations():
    from . import opgroup
    opgroup.interpreter.apply()
    opgroup.lookup.apply()
    opgroup.safe_optimizations.apply()
    opgroup.defined_names.apply()
    # optimizations.context.apply()
    opgroup.stubs.apply()
    opgroup.class_values.apply()
    opgroup.completions.apply()
    opgroup.used_names.apply()
    opgroup.filters.apply()
    opgroup.memo.apply()


# Jedi doesn't consider the indentation at the completion site potentially
# giving the wrong scope at the last line of the scope. This fixes that.
def patch_get_user_context():
    from jedi.api import completion

    def get_user_context(module_context, pos):
        leaf = module_context.tree_node.get_leaf_for_position(pos, include_prefixes=True)

        if leaf.type == "newline":
            parent = leaf.parent
            if parent.get_last_leaf() is leaf and pos[1] < parent.start_pos[1]:
                leaf = leaf.get_next_leaf()

        elif leaf.start_pos > pos or leaf.type == "endmarker":
            if last := leaf.get_previous_leaf():
                 if last.type == "newline":
                    if pos[1] == last.parent.start_pos[1]:
                        leaf = last

        return module_context.create_context(leaf)

    _patch_function(completion.get_user_context, get_user_context)


# This fixes jedi attempting to complete from both stubs and compiled modules,
# which isn't allowed by PEP 484. It's either or.
def patch_convert_values():
    from jedi.inference.gradual.conversion import convert_values

    # This makes ``prefer_stubs`` True by default and False if ``only_stubs`` is True.
    def f(values, only_stubs=False, prefer_stubs=True, ignore_compiled=True):
        if only_stubs:
            prefer_stubs = False
        return convert_values(values, only_stubs=only_stubs, prefer_stubs=prefer_stubs, ignore_compiled=ignore_compiled)
    convert_values = _patch_function(convert_values, f)


# Fix unpickling errors jedi doesn't catch.
def patch_load_from_file_system():
    from parso.cache import _load_from_file_system

    def safe_wrapper(*args, **kw):
        try:
            return _load_from_file_system(*args, **kw)
        except ModuleNotFoundError:
            return None  # Return None so jedi can re-save it.
        
    _load_from_file_system = _patch_function(_load_from_file_system, safe_wrapper)


def patch_NameWrapper_getattr():
    from jedi.inference.names import NameWrapper

    # del NameWrapper.__getattr__

    NameWrapper.public_name = _forwarder("_wrapped_name.public_name")
    NameWrapper.get_public_name = _forwarder("_wrapped_name.get_public_name")
    NameWrapper.string_name = _forwarder("_wrapped_name.string_name")
    NameWrapper.parent_context = _forwarder("_wrapped_name.parent_context")
    # NameWrapper.tree_name = _forwarder("_wrapped_name.tree_name")


def patch_SequenceLiteralValue():
    from jedi.inference.value.iterable import SequenceLiteralValue
    from jedi.inference.gradual.base import GenericClass, _LazyGenericBaseClass
    from textension.utils import falsy_noargs, truthy_noargs
    from jedi.cache import memoize_method

    @memoize_method
    def py__bases__(self: GenericClass):
        ret = []
        bases = list(self._wrapped_value.py__bases__())
        for base in bases:
            add = _LazyGenericBaseClass(self, base, self._generics_manager)
            ret += [add]
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
    TypeVar.get_safe_value = _forwarder("_wrapped_value.get_safe_value")
    TypeVar.is_compiled    = _forwarder("_wrapped_value.is_compiled")
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
    from jedi.inference.value.instance import CompiledInstance
    from jedi.inference.compiled.value import CompiledValue
    from jedi.inference.base_value import Value, NO_VALUES, ValueSet

    from .tools import make_compiled_value

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
    from jedi.inference.compiled.value import CompiledModule
    from jedi.inference.base_value import ValueSet
    from jedi.inference.imports import Importer, import_module

    from .common import Importer_redirects
    from .tools import get_handle, state
    import importlib

    module_cache = state.module_cache

    def follow(self: Importer):
        if module := Importer_redirects.get(".".join(self._str_import_path)):
            return ValueSet((module,))
        return follow_orig(self)

    follow_orig = _patch_function(Importer.follow, follow)

    # Overrides import completion names.
    def completion_names(self, inference_state, only_modules=False):
        if module := Importer_redirects.get(".".join(self._str_import_path)):
            return module.get_submodule_names(only_modules=only_modules)
        return org_completion_names(self, inference_state, only_modules=only_modules)

    org_completion_names = Importer.completion_names
    Importer.completion_names = completion_names

    # Required for builtin module import fallback when Jedi fails.
    def import_module_override(state, name, parent_module, sys_path, prefer_stubs):
        # XXX: This might fetch compiled module which breaks when jedi thinks it's stub.
        # if ret := module_cache.get(name):
        #     return ret

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


# Fixes imports by redirecting them to virtual modules, because:
# 1. bpy/__init__.py is inferred by source text, not its runtime members.
# 2. bpy.app is not a module and relies on a loophole in import resolution.
# 3. _bpy has no spec/loader causing module detection heuristics to fail.
# 4. bpy.ops uses getattr and dynamic modules implying code execution.
# All which are reasons for Jedi completing them partially or not at all.
# So instead of addressing each problem differently, we use virtual modules.
def patch_Importer_follow():
    from jedi.inference.imports import Importer, ValueSet
    from .common import Importer_redirects

    def follow(self: Importer):
        path = ".".join(self._str_import_path)
        if module := Importer_redirects.get(path):
            return ValueSet((module,))
        return follow_orig(self)

    follow_orig = _patch_function(Importer.follow, follow)


# Fixes completion for "import _bpy" and other file-less modules which
# are not listed in sys.builtin_module_names.
def patch_get_builtin_module_names():
    from jedi.inference.compiled.subprocess.functions import get_builtin_module_names
    from builtins import set, map, getattr
    from itertools import repeat, compress
    from operator import and_, not_, contains
    import sys

    builtin = {*sys.builtin_module_names}

    module_names = sys.modules.keys()
    modules = sys.modules.values()

    file  = repeat("__file__")
    false = repeat(False)
    dot   = repeat(".")

    def get_builtin_module_names_p(_):
        # Skip modules with no __file__.
        non_files = map(not_, map(getattr, modules, file, false))
        # Skip modules with dot in their name.
        non_dotted = map(not_, map(contains, module_names, dot))
        # Composite with actual builtin modules.
        return builtin | set(compress(module_names, map(and_, non_files, non_dotted)))

    _patch_function(get_builtin_module_names, get_builtin_module_names_p)


def patch_misc():
    from importlib import _bootstrap_external
    from importlib._bootstrap_external import path_sep, path_separators
    from itertools import starmap, repeat
    from builtins import zip, filter
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
        return join(starmap(rstrip, zip(filter(None, parts), path_seps)))
    _patch_function(_bootstrap_external._path_join, _path_join)


# Patches FakeList to have its 'array_type' class attribute fixed.
def patch_fakelist_array_type():
    from jedi.inference.value.iterable import FakeList
    FakeList.array_type = 'list'


# Patches CompiledValue to support inference and extended py__call.
# Specifically fixes [].copy().
#                              ^
def patch_compiledvalue():
    from jedi.inference.compiled.access import create_access_path
    from jedi.inference.compiled.value import CompiledValue, \
        create_from_access_path, ValueSet, NO_VALUES

    from .modules._bpy_types import RnaValue
    
    import typing
    from typing import GenericAlias, _GenericAlias, Hashable

    is_hashable = Hashable.__instancecheck__
    _AliasTypes = (GenericAlias, _GenericAlias)

    # Try to convert GenericAlias into _GenericAlias, the old type.
    def convert_alias(alias):
        if not isinstance(alias, _AliasTypes):
            return alias
        args = tuple(convert_alias(a) for a in getattr(alias, "__args__"))
        origin = getattr(typing, alias.__origin__.__name__.capitalize(), alias)
        return _GenericAlias(origin, args)

    # This intercepts CompiledValue.py__call__ to check if it's an RNA class
    # or instance and returns an RnaResolver. This makes bpy annotations work
    # as RNA instances instead of classes.
    def py__call__extended(self: CompiledValue, arguments):
        obj = self.access_handle.access._obj
        if hasattr(obj, "bl_rna"):
            return ValueSet((RnaValue(obj, self.parent_context).instance,))

        elif isinstance(obj, GenericAlias):
            a = create_access_path(self.inference_state, convert_alias(obj))
            return create_from_access_path(self.inference_state, a).execute_annotation()

        # Obviously non-hashable objects generally aren't useful to lookup.
        if not is_hashable(obj):
            return NO_VALUES

        else:
            ret = NO_VALUES
        return ret

    # This patch is meant to intercept py__call__ and force jedi to infer
    # objects it doesn't handle out-of-the-box.
    org_py__call__ = CompiledValue.py__call__
    def py__call__(self, arguments):
        return py__call__extended(self, arguments) or org_py__call__(self, arguments)

    CompiledValue.py__call__ = py__call__


# Patches handle-to-CompiledValue creation to intercept compiled objects.
# Needed as part of RNA inference integration.
def patch_create_cached_compiled_value():
    from jedi.inference.compiled.value import (CompiledModule,
                                               CompiledValue,
                                               inference_state_function_cache,
                                               create_cached_compiled_value,
                                               _normalize_create_args)
    from bpy_types import RNAMeta, StructRNA
    from .modules._bpy_types import get_rna_value

    is_compiled_value = CompiledValue.__instancecheck__
    bpy_types = (RNAMeta, StructRNA, bpy.props._PropertyDeferred)
    from .common import CompiledModule_redirects

    @_normalize_create_args
    @inference_state_function_cache()
    def create_cached_compiled_value_p(state, handle, context):
        obj = handle.access._obj

        if context:
            assert not is_compiled_value(context)
            if isinstance(obj, bpy_types):
                return get_rna_value(obj, context)
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


# Patch strings.complete_dict because of a possible AttributeError.
def patch_complete_dict():
    from jedi.api.strings import complete_dict
    def safe_wrapper(*args, **kw):
        try:
            return complete_dict(*args, **kw)
        # ``before_bracket_leaf`` can be None, but Jedi doesn't test this.
        except AttributeError:
            return []
    complete_dict = _patch_function(complete_dict, safe_wrapper)
