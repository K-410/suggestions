# This adds fixes for jedi to outright work.


from .tools import state, _descriptor_overrides, _value_overrides, \
    _patch_function, _get_unbound_super_method, _copy_function

from .modules._bpy_types import get_rna_value, patch_AnonymousParamName_infer
from .modules._mathutils import apply_mathutils_overrides
from .imports import fix_bpy_imports

import bpy

def apply():
    _apply_optimizations()
    _apply_patches()


def _apply_patches():
    # patch_hide_shit_for_debugging()  # XXX: For debuggin

    fix_bpy_imports()
    apply_mathutils_overrides()

    patch_NameWrapper_getattr()
    patch_Completion_complete_inherited()
    patch_Value_py__getattribute__alternatives()
    patch_fakelist_array_type()
    patch_import_resolutions()
    # patch_Importer_follow()
    patch_AnonymousParamName_infer()
    patch_get_builtin_module_names()
    patch_compiledvalue()  # XXX: Not having this means tuple[X] isn't subscriptable.
    # patch_ClassMixin()  # XXX: This loads stub modules on startup.
    # patch_misc()  # XXX: This is very experimental.
    patch_StubFilter()
    patch_is_pytest_func()
    # patch_AbstractInstanceValue_py__getitem()  # XXX: Not relevant when using stub modules.
    # patch_SelfAttributeFilter_filter_self_names()  # XXX: Clean this up.
    patch_paths_from_list_modifications()

    # from .resolvers import init_lookups
    # init_lookups()

    patch_create_cached_compiled_value()


def _apply_optimizations():
    pass
    from .optimizations import interpreter
    interpreter.apply()


import gpu

_fallbacks = {
    "GPUShader": gpu.types.GPUShader
}


# Fixes stack overflow of NameWrapper.__getattr__ during debugging.
def patch_NameWrapper_getattr():
    from jedi.inference.names import NameWrapper

    _getattribute = object.__getattribute__
    del NameWrapper.__getattr__

    def __getattribute__(self, name):
        try:
            return _getattribute(self, name)
        except AttributeError:
            return getattr(_getattribute(self, "_wrapped_name"), name)
        
    NameWrapper.__getattribute__ = __getattribute__


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


def patch_hide_shit_for_debugging():
    from dev_utils import hide

    from parso.python.tree import _StringComparisonMixin
    from jedi.inference.base_value import ValueSet
    from jedi.debug import dbg, increase_indent_cm
    from textension.utils import _forwarder

    hide(_StringComparisonMixin.__eq__)
    hide(dbg)
    # hide(ValueSet.py__getattribute__)
    # hide(ValueSet.from_sets)
    # hide(ValueSet._from_frozen_set)
    hide(increase_indent_cm)
    del ValueSet.__bool__
    ValueSet.__len__ = _forwarder("_set.__len__")
    ValueSet.__iter__ = _forwarder("_set.__iter__")


# Fallback for py__getattribute__.
def patch_Value_py__getattribute__alternatives():
    from jedi.inference.base_value import Value, NO_VALUES, ValueSet
    from jedi.inference.compiled.value import CompiledValue
    from jedi.inference.value.instance import CompiledInstance
    from .tools import make_compiled_value

    def py__getattribute__alternatives(self: Value, name_or_str):
        if obj := _fallbacks.get(name_or_str):
            return ValueSet((make_compiled_value(obj, self.as_context()),))
        
        elif isinstance(self, CompiledInstance):
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

        print("no fallbacks found for", repr(name_or_str))
        return NO_VALUES
    
    Value.py__getattribute__alternatives = py__getattribute__alternatives


# Jedi breaks if there's ill-formed code related to sys path modifications.
# Think ``sys.path.append(foo, bar)``, forgetting a set of parentheses.
# Obviously such code won't run, but it renders Jedi unusable as it raises
# an exception and prevents further completions in the source from working.
def patch_paths_from_list_modifications():
    from jedi.inference.sys_path import _paths_from_list_modifications

    def _safe_wrapper(*args, **kw):
        # Materialize now so this generator can blow up at the call site.
        try:
            return list(orig_fn(*args, **kw))

        # We don't care about the exception.
        except:
            return ()

    orig_fn = _copy_function(_paths_from_list_modifications)
    _patch_function(_paths_from_list_modifications, _safe_wrapper)


# Patches filter_self_names to detect bpy.props properties.
# TODO: This needs a LOT of cleanup.
def patch_SelfAttributeFilter_filter_self_names():
    from jedi.inference.value.instance import SelfAttributeFilter
    
    from jedi.inference.syntax_tree import infer_node
    from jedi.inference.compiled.value import CompiledValue
    from .resolvers import PropResolver

    def _filter_self_names(self, names):
        # class_value = self.parent_context.get_value()
        # for base in class_value.py__bases__():
        #     for v in base.infer():
        #         if v.is_compiled():
        #             obj = v.access_handle.access._obj
        #             if hasattr(obj, "bl_rna"):


        # Check if the name originates from a bpy.props annotation assignment.
        # Requires underlying object to already be overridden by PropResolver.
        def check_is_bpy_prop_origin(name):
            parent = name.parent
            if parent.type == "expr_stmt":
                ret = parent.children[1]  # The right-hand side
                if ret.type == "annassign":
                    # r, = infer_node(self.parent_context, ret.children[1])
                    # return isinstance(r.access_handle.access._obj, PropResolver)

                    # XXX: Can fail because 'r' has no access_handle member.
                    try:
                        r, = infer_node(self.parent_context, ret.children[1])
                        return isinstance(r.access_handle.access._obj, PropResolver)
                    except:
                        pass
            return False

        def check_is_property(name):
            parent = name.parent
            if parent.type == "funcdef" and parent.parent.type == "decorated":
                tmp = parent.get_previous_sibling()
                # assert tmp.type == "decorator"
                if tmp.type == "decorator":
                    r = tmp.get_last_leaf().get_previous_sibling().get_last_leaf()
                    # XXX: Can fail because infer_node returns nothing.
                    try:
                        o, = infer_node(self.parent_context, r)
                    except:
                        pass
                    else:
                        if isinstance(o, CompiledValue):
                            try:
                                return issubclass(o.access_handle.access._obj, property)
                            except:
                                pass
            return False                                

        is_property = False
        for name in names:
            is_property = check_is_bpy_prop_origin(name) or check_is_property(name)
            if is_property:
                break

        for name in names:
            trailer = name.parent
            if trailer.type == 'trailer' \
                    and len(trailer.parent.children) == 2 \
                    and trailer.children[0] == '.':
                if name.is_definition() and self._access_possible(name):
                    # TODO filter non-self assignments instead of this bad
                    #      filter.
                    if self._is_in_right_scope(trailer.parent.children[0], name):
                        if not is_property:
                            yield name

    SelfAttributeFilter._filter_self_names = _filter_self_names


# Patch ``_is_pytest_func`` to return False. Otherwise we can't define a 
# function called ``test`` without jedi thinking it's somehow pytest related.
def patch_is_pytest_func():
    from jedi.plugins.pytest import _is_pytest_func
    _patch_function(_is_pytest_func, lambda *_: False)


# Patches AbstractInstanceValue to make its py__getitem__ support annotations
# for underlying compiled values. Required when using compiled builtins module.
# Specifically solves typevar inference of lists
#
# a: list[list[int]] = []
# a[0][0].
#         ^ completes integer instances.
# def patch_AbstractInstanceValue_py__getitem():
#     from jedi.inference.value.instance import AbstractInstanceValue

#     py__getitem__org = AbstractInstanceValue.py__getitem__

#     def py__getitem__(self, index_value_set, contextualized_node):
#         if self.class_value.is_compiled():
#             index, = index_value_set
#             # XXX: This assumes obj is an integer. Won't work for dicts.
#             index = index._compiled_value.access_handle.access._obj
#             if hasattr(self.class_value, "_generics_manager"):
#                 try:
#                     typevar, = self.class_value.get_generics()[index]
#                 except:
#                     ret = self.class_value._generics_manager.get_index_and_execute(index)
#                     return ret
#                     # typevar, = self.class_value.py__getitem__(index, None)
#                     pass
#                 ret = typevar.py__call__(None)
#                 return ret

#         return py__getitem__org(self, index_value_set, contextualized_node)

#     AbstractInstanceValue.py__getitem__ = py__getitem__


# Patches Jedi's import resolutions to allow completing sub-modules assigned
# either at runtime or from built-in, file-less modules. This is required for
# import completions of both compiled and mixed file/runtime modules like bpy,
# gpu, mathutils, etc. Names are obtained from compiled modules using safety
# principles similar to how Jedi does things in getattr_static.
def patch_import_resolutions():
    from jedi.inference.imports import Importer, import_module as org_import_module
    from jedi.inference.compiled.access import create_access
    from jedi.inference.compiled.value import CompiledModule
    from jedi.inference.names import SubModuleName
    from jedi.inference.base_value import ValueSet

    from .tools import _filter_modules, get_handle
    from .common import Importer_redirects

    from importlib.util import find_spec
    from importlib import import_module
    from itertools import repeat
    from types import ModuleType

    from sys import modules
    from os.path import dirname
    from os import listdir
    import _bpy

    org_follow = Importer.follow

    def follow(self: Importer):
        if module := Importer_redirects.get(".".join(self._str_import_path)):
            return ValueSet((module,))
        ret = org_follow(self)
        return ret

    Importer.follow = follow

    # Overrides import completion names.
    def completion_names(self, inference_state, only_modules=False):
        if module := Importer_redirects.get(".".join(self._str_import_path)):
            return module.get_submodule_names(only_modules=only_modules)
        return org_completion_names(self, inference_state, only_modules=only_modules)

    org_completion_names = Importer.completion_names
    Importer.completion_names = completion_names

    # Required for builtin module import fallback when Jedi fails.
    def import_module_override(state, name, parent_module, sys_path, prefer_stubs):
        ret = org_import_module(state, name, parent_module, sys_path, prefer_stubs)
        if not ret:
            try:
                module = import_module(".".join(name))
                access_handle = create_access(state, module)
                ret = ValueSet((CompiledModule(state, access_handle),))
            except ImportError:
                pass
        return ret

    org_import_module = _patch_function(org_import_module, import_module_override)

    module_getattr = ModuleType.__getattribute__
    module_dict = ModuleType.__dict__["__dict__"].__get__
    module_dir = ModuleType.__dir__

    # SubModuleNames for CompiledModules are fetched once, then kept in cache.
    sub_modules_cache = {}

    # These don't have sub modules. Skip them.
    excludes = {"builtins", "typing"}

    # For all other compiled modules.
    def get_submodule_names(self: CompiledModule, only_modules=True):
        if self in sub_modules_cache:
            return sub_modules_cache[self]

        result = []
        name = self.py__name__()

        if name not in excludes:
            m = self.access_handle.access._obj
            assert isinstance(m, ModuleType)
            exports = set()

            # It's possible we're passing non-module objects.
            if isinstance(m, ModuleType):
                exports.update(module_dir(m) + list(module_dict(m)))

                # ``__all__`` could be anything.
                try:
                    exports.update(module_getattr(m, "__all__"))
                except:
                    pass

            else:
                exports.update(object.__dir__(m))

            # If the module is a file, find sub modules.
            for path in module_getattr(m, "__path__", ()):
                exports.update(_filter_modules(listdir(path)))

            names = []
            for e in filter(str.__instancecheck__, exports):

                # Skip double-underscored names.
                if e[:2] == "__" == e[-2:]:
                    continue

                if only_modules:
                    full_name = f"{name}.{e}"
                    if full_name not in modules:
                        try:
                            assert find_spec(full_name)
                        except (ValueError, AssertionError, ModuleNotFoundError):
                            continue
                names += [e]

            result = list(map(SubModuleName, repeat(self.as_context()), names))
        return sub_modules_cache.setdefault(self, result)


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

    follow_orig = Importer.follow
    Importer.follow = follow


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

    # 3789   22.337    0.006   39.381    0.010 XXX: Default
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
    from jedi.inference.compiled.access import create_access_path, create_access
    from jedi.inference.value import CompiledInstance
    from jedi.inference.lazy_value import LazyKnownValue
    from jedi.inference.compiled.value import CompiledValue, \
        create_cached_compiled_value, create_from_access_path, ValueSet, Value, NO_VALUES

    from .resolvers import RnaResolver, iter_lookup, subscript_lookup, \
        restype_lookup, attr_lookup, _AliasTypes, func_map
    
    import typing
    from typing import GenericAlias, _GenericAlias, Hashable

    is_hashable = Hashable.__instancecheck__

    def make_compiled(self: Value, obj):
        state = self.inference_state
        return create_cached_compiled_value(state, create_access(state, obj), self.parent_context)

    def make_instance(self, obj, arguments=None):
        value = make_compiled(self, obj)
        return CompiledInstance(self.inference_state, self.parent_context, value, arguments)

    def get_return_annotation(self: CompiledValue):
        ret = self.access_handle.get_return_annotation()
        if not ret:
            # Guard against unhashable types.
            try:
                if obj := restype_lookup.get(self.access_handle.access._obj):
                    return create_access_path(self.inference_state, obj)
            except:
                pass
        return ret
    CompiledValue.get_return_annotation = get_return_annotation

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
            return ValueSet([make_instance(self, RnaResolver(obj, static=True), arguments)])

        elif isinstance(obj, GenericAlias):
            a = create_access_path(self.inference_state, convert_alias(obj))
            return create_from_access_path(self.inference_state, a).execute_annotation()

        # Obviously non-hashable objects generally aren't useful to lookup.
        if not is_hashable(obj):
            return NO_VALUES

        if obj in restype_lookup:
            # print(green("restype in lookup"))
            restype = restype_lookup[obj]

            # XXX: Jedi doesn't understand the new GenericAlias types as return values
            # XXX: from CompiledValue.
            if isinstance(restype, GenericAlias):
                restype = convert_alias(restype)
                # print(red("convert alias"), restype)
                # return ValueSet([create_instance(self, restype)])
            # NOTE: This only works with _GenericAlias, NOT GenericAlias.
            # NOTE: This means List and Tuple, and NOT list and tuple.
            if isinstance(restype, _GenericAlias):
                # ret = ValueSet([create_compiled(self, restype)])
                a = create_access_path(self.inference_state, restype)
                return create_from_access_path(self.inference_state, a).execute_annotation()

            else:
            #     print("make instance")
                ret = ValueSet([make_instance(self, restype)])
        elif obj in func_map:
            # print("in func_map", func_map[obj])
            ret = ValueSet((make_compiled(self, func_map[obj]),))
        else:
            ret = NO_VALUES
        return ret

    # This patch is meant to intercept py__call__ and force jedi to infer
    # objects it doesn't handle out-of-the-box.
    org_py__call__ = CompiledValue.py__call__
    def py__call__(self, arguments):
        return py__call__extended(self, arguments) or org_py__call__(self, arguments)
    CompiledValue.py__call__ = py__call__



    # Patch CompiledValue.py__iter__ to resolve iterators jedi can't.
    def py__iter__extended(self: CompiledValue):
        obj = self.access_handle.access._obj
        if obj in iter_lookup:
            value = make_instance(self, iter_lookup[obj])
            return [LazyKnownValue(value)]
        return []

    org_py__iter__ = CompiledValue.py__iter__
    def py__iter__(self, contextualized_node=None):
        # print(yellow("py_iter"), self)
        ret = list(org_py__iter__(self, contextualized_node))
        if not ret:
            ret = py__iter__extended(self)
        # if not ret:
        #     print("  nothing")
        # else:
        #     print("  found:", ret)
        return ret
    CompiledValue.py__iter__ = py__iter__


    def py__getitem__extended(self: CompiledValue):
        obj = self.access_handle.access._obj
        if is_hashable(obj) and obj in subscript_lookup:
            value = make_instance(self, subscript_lookup[obj])
            return ValueSet([value])
        return NO_VALUES

    org_py__getitem__ = CompiledValue.py__getitem__
    def py__getitem__(self: CompiledValue, index_value_set, contextualized_node):
        # print(yellow("py_getitem"), self)
        ret = org_py__getitem__(self, index_value_set, contextualized_node)
        if not ret:
            ret = py__getitem__extended(self)
        # if not ret:
        #     print("  nothing")
        # else:
        #     print("  found:", ret)
        return ret
    CompiledValue.py__getitem__ = py__getitem__




    org_py__getattribute__ = CompiledInstance.py__getattribute__
    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        ret = org_py__getattribute__(self, name_or_str, name_context=name_context,
                                     position=position, analysis_errors=analysis_errors)
        # print(yellow("py_getattr"), name_or_str, yellow("on"), self)
        if ret:
            v, *_ = ret
            if not v.is_instance():
                obj = v.access_handle.access._obj
                if obj in attr_lookup:
                    # print("creating instance")
                    obj = attr_lookup[obj]
                    ret = make_instance(self, obj)
                    # print("now", ret)
                    ret = ValueSet([ret])
                # else:
                #     print("not instance, not in lookup")
            # print("  found:", ret)
        # else:
        #     print("  nothing")
        return ret
    CompiledInstance.py__getattribute__ = py__getattribute__


# XXX: This loads stub modules on startup.
# Patches ClassMixin to support compiled value filters.
# Patches get_filters to support metaclasses.
# Not having this causes jedi to recurse to infinity or a stack overflow when
# trying to get filters on compiled values.
# This patch also optimizes the function by caching filters on the tree node.
def patch_ClassMixin():
    from jedi.inference.value.klass import ClassMixin, ClassFilter, ClassValue
    from jedi.inference.compiled import builtin_from_name, CompiledValueFilter

    # TODO: Move this to tools or something.
    type_filter = ClassFilter(builtin_from_name(state, "type"), None, None)

    def get_filters_real(
            self: ClassMixin | ClassValue,
            origin_scope=None,
            is_instance=False,
            include_metaclasses=True,
            include_type_when_class=True):

        # Jedi's metaclass search is broken. This fixes it.
        if include_metaclasses:
            for cls in self.get_metaclasses():
                yield ClassFilter(self, node_context=cls.as_context(), origin_scope=origin_scope)

        if not is_instance and include_type_when_class:
            yield type_filter

        # ``is_wrapped`` is true when the value is tied to a compiled object.
        is_wrapped = hasattr(self, "access_handle")
        for cls in self.py__mro__():
            if cls.is_compiled():
                if is_wrapped:
                    yield CompiledValueFilter(self.inference_state, self, is_instance)
                else:
                    yield from cls.get_filters(is_instance=is_instance)
            else:
                f = ClassFilter(self,
                                  node_context=cls.as_context(),
                                  origin_scope=origin_scope)
                f._is_instance = is_instance
                yield f
                # yield ClassFilter(self,
                #                   node_context=cls.as_context(),
                #                   origin_scope=origin_scope,
                #                   is_instance=is_instance)

    def get_filters(self: ClassMixin | ClassValue, *args, **kw):
        return get_filters_real(self, *args, **kw)

    ClassMixin.get_filters = get_filters


# Patch StubFilter to lookup names starting with underscore using the stub's
# compiled counterpart, if it currently exists, as fallback.
# This specifically fixes completing sys._getframe, which jedi simply filters
# out in the default implementation.
def patch_StubFilter():
    from jedi.inference.gradual.stub_value import StubFilter
    from types import ModuleType
    from builtins import set

    super_meth = _get_unbound_super_method(StubFilter, "_is_name_reachable")
    get_dict = ModuleType.__dict__["__dict__"].__get__
    is_str = str.__instancecheck__

    # This cache stores the string names defined in compiled modules and
    # persists for the duration of the application for performance reasons.
    private_names_lookup_cache = {}

    # Omit import aliases (eg. import X as _X) from stub modules.
    def is_import_alias(name):
        if definition := name.get_definition():
            if definition.type in {"import_from", "import_name"}:
                if name.parent.type not in {"import_as_name", "dotted_as_name"}:
                    return True
        return False

    def _is_name_reachable_p(self: StubFilter, name):
        # TODO: StubFilters are static so caching could be done here.

        if not super_meth(self, name) or is_import_alias(name):
            return False

        value = name.value

        # Allow *some* private names from stub modules, if they exist
        # on the compiled module. Assuming a compiled module even exists.
        if value[0] is "_" and (value[:2] != "__" != value[-2:]):
            if self in private_names_lookup_cache:
                return value in private_names_lookup_cache[self]

            ret = []
            values = self.parent_context._value.non_stub_value_set
            if self.parent_context.is_compiled():
                for v in values:
                    module = v.access_handle.access._obj
                    if isinstance(module, ModuleType):
                        ret = [k for k in get_dict(module) if is_str(k)]
            else:
                ret = (n.string_name for v in values for n in next(v.get_filters()).values())
            return value in private_names_lookup_cache.setdefault(self, set(ret))

        return True


    StubFilter._is_name_reachable = _is_name_reachable_p


# Patches handle-to-CompiledValue creation to intercept compiled objects.
# Needed as part of RNA inference integration.
def patch_create_cached_compiled_value():
    from jedi.inference.compiled.value import (CompiledModule,
                                               CompiledValue,
                                               inference_state_function_cache,
                                               create_cached_compiled_value,
                                               _normalize_create_args)
    from bpy_types import RNAMeta, StructRNA

    is_compiled_value = CompiledValue.__instancecheck__
    bpy_types = (RNAMeta, StructRNA, bpy.props._PropertyDeferred)
    from .common import CompiledModule_redirects

    @_normalize_create_args
    @inference_state_function_cache()
    def create_cached_compiled_value_p(state, handle, context):
        obj = handle.access._obj

        if context:
            assert not is_compiled_value(context)
            if isinstance(obj, type(callable)):
                pass
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
