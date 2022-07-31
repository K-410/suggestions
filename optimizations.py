"""
This module contains optimizations for jedi and parso.

Version compatibility:
    See _version_compat_jedi / _version_compat_parso
"""

_version_compat_jedi = '0.18.1'
_version_compat_parso = '0.8.3'



from jedi.api import Script
from jedi.api.project import Project
from jedi.api.environment import InterpreterEnvironment
from jedi.api.interpreter import MixedModuleContext
from jedi.api.completion import Completion
from jedi.inference import InferenceState
from jedi.inference.value.module import ModuleValue

from jedi import settings
from pathlib import Path

from functools import lru_cache


def _clear_caches(_caches=[], _clear=dict.clear) -> None:
    """
    Internal.
    Evict all caches passed to, or obtained by, _register_cache().
    """
    for cache in _caches:
        _clear(cache)
    return None

def count_cached():
    n = 0
    items = [_clear_caches.__defaults__[0]]
    while items:
        elem = items.pop()
        if isinstance(elem, dict):
            items.extend(elem.values())
            # print("extending dict...")
        else:
            try:
                # iter(elem)
                items.extend(elem)
                # print("extending iterable...", elem)
            except:
                pass
                # print("not iterable", elem)
        n += 1
    return n


def _register_cache(cache: dict = None) -> dict | None:
    if cache is None:
        cache = {}
    assert isinstance(cache, dict), f"Expected dict or None, got {type(cache)}"
    _clear_caches.__defaults__[0].append(cache)
    return cache


def _patch(dst_fn, src_fn) -> None:
    """
    Replaces the destination function's code object with source function.
    """
    dst_fn.__code__ = src_fn.__code__
    return None


def _is_patched(fn) -> bool:
    """
    Returns whether a function was patched by this module.
    """
    return fn.__code__.co_filename == __file__


def _inject_const(**kwargs):
    """
    Internal.
    @inject_const(string="Hello World", none_varialbe=None)
    def func():
        print("const string")
        assert "const none_variable" is None
    """
    def _injector(func):
        for attr, obj in kwargs.items():
            *consts, = func.__code__.co_consts
            consts[consts.index(f"const {attr}")] = obj
            func.__code__ = func.__code__.replace(co_consts=tuple(consts))
        return func
    return _injector


def setup2():
    import jedi
    import parso
    assert jedi.__version__ == _version_compat_jedi
    assert parso.__version__ == _version_compat_parso

    optimize_get_parent_scope()
    optimize_get_cached_parent_scope()  # Call after optimize_get_parent_scope
    optimize_platform_system()
    optimize_abstractfilter_filter()
    optimize_name_is_definition()
    optimize_get_used_names()
    optimize_valuewrapperbase_name()
    optimize_shadowed_dict()
    optimize_static_getmro()
    optimize_getattr_static()
    optimize_check_flows()
    optimize_parsertreefilter_filter()  # Call after optimize_cached_parent_scope
    optimize_classfilter_filter()       # Call after optimize_parsertreefilter_filter
    optimize_leaf_end_pos_descriptor()
    optimize_get_definition_names()
    optimize_selfattributefilter_filter()
    optimize_split_lines()
    optimize_basenode_end_pos_descriptor()
    optimize_abstractusednamesfilter_values()  # Call after optimize_get_definition_names
    optimize_fuzzy_match()
    optimize_filter_names()
    optimize_valuewrapperbase_getattr()
    optimize_lazyvaluewrapper_wrapped_value()


class Interpreter2(Script):
    _orig_path = None
    path = Path("~")  # Set an invalid path so Jedi won't recurse it.
    _project = Project(path)
    _inference_state = InferenceState(_project, InterpreterEnvironment(), None)
    _inference_state.allow_descriptor_getattr = True
    namespaces = []
    
    def __init__(self, code: str, _=[]):
        self._module_node = self._inference_state.grammar.parse(
            code=code, diff_cache=settings.fast_parser, cache_path=settings.cache_directory)
        self._code_lines = lines = code.splitlines(True)
        self._code = code
        try:
            if code[-1] is "\n":
                lines.append("")
        except IndexError:
            lines.append("")

    @lru_cache
    def _get_module_context(self):
        tree_module_value = ModuleValue(
            self._inference_state, self._module_node, file_io=None,
            string_names=('__main__',), code_lines=self._code_lines)
        return MixedModuleContext(tree_module_value, self.namespaces)

    def complete_unsafe(self, line, column, *, fuzzy=False):
        """
        Same as 'complete()', but with required line/column and no checks/debug.
        """
        return Completion(
            self._inference_state, self._get_module_context(),
            self._code_lines, (line, column), self.get_signatures, fuzzy=fuzzy
        ).complete()


def optimize_platform_system():
    """
    Replace platform.system with sys.platform.

    The former is runtime computed, which is generally slow and useless.
    The latter is returns a constant string from when python was compiled.
    """
    import sys
    import platform
    platform._system = platform.system  # Backup the original function
    platform.system = sys.platform.replace("win32", "windows").title


def optimize_abstractfilter_filter():
    # @_inject_const(cache={}, append=list.append)
    def _filter(self, names):
        if (until_position := self._until_position) is not None:
            return [n for n in names if n.start_pos < until_position]
        return names

    from jedi.inference.filters import AbstractFilter
    _patch(AbstractFilter._filter, _filter)


def optimize_name_is_definition():
    from parso.python.tree import Name, _GET_DEFINITION_TYPES

    @_inject_const(_GET_DEFINITION_TYPES=_GET_DEFINITION_TYPES)
    def is_definition(self, include_setitem=False):
        node = self.parent
        node_type = node.type
        if node_type in {'funcdef', 'classdef'}:
            if self == node.name:
                return node
        elif node_type == 'except_clause':
            if self.get_previous_sibling() == 'as':
                return node.parent
        while node is not None:
            if node.type in "const _GET_DEFINITION_TYPES":
                if self in node.get_defined_names(include_setitem):
                    return node
                break
            node = node.parent
        return None
    Name.is_definition = is_definition
    _patch(Name.is_definition, is_definition)


def optimize_get_used_names():
    """
    Replaces parso.python.tree.Module.get_used_names
    """
    @_inject_const(pop=list.pop, hasattr=hasattr, extend=list.extend, append=list.append)
    def get_used_names(self):
        if (used_names := self._used_names) is not None:
            return used_names
        pool = [self]
        used_names = {}
        while pool:
            node = "const pop"(pool)
            if "const hasattr"(node, "children"):
                "const extend"(pool, node.children)
            elif node.type == "name":
                if (value := node.value) in used_names:
                    "const append"(used_names[value], node)
                else:
                    used_names[value] = [node]
        self._used_names = used_names
        return used_names
    from parso.python.tree import Module
    _patch(Module.get_used_names, get_used_names)


def optimize_valuewrapperbase_name():
    """
    Replaces _ValueWrapperBase.name, whose design uses import and global
    lookup on each access. The optimization here uses a mutating descriptor
    to do the first runtime lookup and closes them into cells.
    """
    # A mutating property descriptor that binds runtime imports (bad design).
    # This makes imports into cell variables, which loads identical to the
    # LOAD_FAST bytecode instruction, and inlines exception handling.
    # from jedi.inference.base_value import _ValueWrapperBase
    # from jedi import inference
    from jedi.inference import utils, names, compiled, base_value

    @_inject_const(
        ValueName=names.ValueName,
        CompiledValueName=compiled.CompiledValueName,
        UncaughtAttributeError=utils.UncaughtAttributeError)
    def name(self):
        try:
            if (name := self._wrapped_value.name.tree_name) is not None:
            # if (name := type(self)._wrapped_value.fget(self).name.tree_name) is not None:
                return "const ValueName"(self, name)
            return "const CompiledValueName"(self, name)
        except AttributeError as e:
            raise "const UncaughtAttributeError"(e) from e
    base_value._ValueWrapperBase.name = property(name)


def optimize_basenode_end_pos_descriptor():
    from parso.tree import BaseNode

    class Cache(dict):
        def __missing__(self, key):
            self[key] = ret = key.children[-1].end_pos
            return ret
    # TODO: Does this cache need to be cleared?
    BaseNode.end_pos_cache = Cache()
    BaseNode.end_pos = property(BaseNode.end_pos_cache.__getitem__)


def optimize_shadowed_dict():
    """
    Replaces 'getattr_static._shadowed_dict'
    """
    from jedi.inference.compiled import getattr_static
    from types import GetSetDescriptorType

    @_inject_const(cache=_register_cache(),
                  getmro=type.__dict__['__mro__'].__get__,
                  getdict=type.__dict__["__dict__"].__get__,
                  sentinel=getattr_static._sentinel,
                  getset=GetSetDescriptorType)
    def _shadowed_dict(klass):
        if klass in "const cache":
            return "const cache"[klass]
        try:
            for entry in "const getmro"(klass):
                a = "const getdict"(entry)
                if "__dict__" in a:
                    class_dict = a["__dict__"]
                    
                    if not (type(class_dict) is "const getset" and \
                            class_dict.__name__ == "__dict__" and \
                            class_dict.__objclass__ is entry):
                        "const cache"[klass] = class_dict
                        return class_dict

        except:
            from jedi.debug import warning
            warning('mro of %s returned %s, should be a tuple' % (klass, "const getmro"(klass)))
            "const cache"[klass] = "const sentinel"
            return "const sentinel"

    _patch(getattr_static._shadowed_dict, _shadowed_dict)


def optimize_get_definition_names():
    """
    Optimizes _get_definition_names in 'jedi/inference/filters.py'
    """
    
    @_inject_const(cache=_register_cache())  # TODO: Figure out optimal eviction policy
    def _get_definition_names(parso_cache_node, used_names, name_key):
        try:
            return "const cache"[parso_cache_node][name_key]
        except:  # Assume KeyError, but skip the error lookup
            try:
                ret = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
            except:  # Unlikely. Assume KeyError, but skip the error lookup
                ret = ()
            if parso_cache_node is not None:
                try:
                    "const cache"[parso_cache_node][name_key] = ret
                except:  # Unlikely. Assume KeyError, but skip the error lookup
                    "const cache"[parso_cache_node] = {name_key: ret}
            return ret

    from jedi.inference import filters
    _patch(filters._get_definition_names, _get_definition_names)


def optimize_abstractusednamesfilter_values():
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from jedi.inference import filters

    cache = next(const for const in filters._get_definition_names.__code__.co_consts if isinstance(const, dict))

    @_inject_const(append=list.append, cache=cache)
    def values(self):
        _filter = self._filter
        parso_cache_node = self._parso_cache_node
        used_names = self._used_names
        # print(parso_cache_node)
        #     return self._convert_names("const names_cache"[key])
        # breakpoint()
        # print(hash())
        ret = []

        if parso_cache_node is not None:
            try:
                node_cache = "const cache"[parso_cache_node]
            except:  # Assume KeyError
                node_cache = "const cache"[parso_cache_node] = {}

            # XXX: Copy of optimized _get_definition_names, but with node cache outside loop
            for name_key in used_names:
                try:
                    tmp = node_cache[name_key]
                except:  # Assume KeyError
                    tmp = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
                    node_cache[name_key] = tmp
                for name in _filter(tmp):
                    "const append"(ret, name)

        else:
            for name_key in used_names:
                tmp = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
                for name in _filter(tmp):
                    "const append"(ret, name)
        # "const names_cache"[key] = ret
        # if (key := (parso_cache_node, tuple(used_names))) in "const names_cache":
        #     return "const names_cache"[key]
        # ret = "const names_cache"[key] = self._convert_names(ret)
        # print(self._convert_names.__func__)
        return self._convert_names(ret)
    _patch(_AbstractUsedNamesFilter.values, values)


def optimize_static_getmro():
    from jedi.inference.compiled import getattr_static as getattr_static_module
    from typing import Iterable

    @_inject_const(
        cache = _register_cache(),  # TODO: Figure out optimal eviction policy
        is_iterable_instance=Iterable.__instancecheck__,
        getmro=type.__dict__['__mro__'].__get__)
    def _static_getmro(klass):
        try:
            return "const cache"[klass]
        except:  # Assume KeyError
            mro = "const getmro"(klass)
            if "const is_iterable_instance"(mro):
                "const cache"[klass] = mro
                return mro
            # There are unfortunately no tests for this, I was not able to
            # reproduce this in pure Python. However should still solve the issue
            # raised in GH #1517.
            from jedi.debug import warning
            warning('mro of %s returned %s, should be a tuple' % (klass, mro))
            "const cache"[klass] = ()
            return ()
    _patch(getattr_static_module._static_getmro, _static_getmro)      


def optimize_getattr_static():
    from jedi.inference.compiled import getattr_static as getattr_static_module
    from types import MemberDescriptorType

    # Not really necessary, already modifying code objects, importing functions still works.
    assert _is_patched(getattr_static_module._shadowed_dict), "_shadowed_dict was never modified"
    assert _is_patched(getattr_static_module._static_getmro), "_static_getmro was never modified"

    @_inject_const(
        cache=_register_cache(),  # TODO: Figure out optimal eviction policy
        _shadowed_dict=getattr_static_module._shadowed_dict,
        _check_class=getattr_static_module._check_class,
        _safe_hasattr=getattr_static_module._safe_hasattr,
        _safe_is_data_descriptor=getattr_static_module._safe_is_data_descriptor,
        _static_getmro=getattr_static_module._static_getmro,
        dict_get=dict.get,
        obj_get=object.__getattribute__,
        is_type=type.__instancecheck__,
        _sentinel=getattr_static_module._sentinel,
        MemberDescriptorType=MemberDescriptorType)
    def getattr_static(obj, attr, default=getattr_static_module._sentinel):
        try:
            ret = "const cache"[obj, attr]
            __import__("dev_utils").count_calls(1)
            return ret
        except:  # Assume KeyError
            __import__("dev_utils").count_calls(2)
            instance_result = "const _sentinel"
            
            # if not isinstance(obj, type):
            if not "const is_type"(obj):
                klass = type(obj)
                dict_attr = "const _shadowed_dict"(klass)
                if dict_attr is "const _sentinel" or type(dict_attr) is "const MemberDescriptorType":
                    try:
                        instance_result = "const dict_get"("const obj_get"(obj, "__dict__"), attr, "const _sentinel")
                    except:  # Assume AttributeError
                        pass
            else:
                klass = obj

            klass_result = "const _check_class"(klass, attr)

            if instance_result is not "const _sentinel" and klass_result is not "const _sentinel":
                if "const _safe_hasattr"(klass_result, '__get__') and "const _safe_is_data_descriptor"(klass_result):
                    # A get/set descriptor has priority over everything.
                    "const cache"[obj, attr] = klass_result, True
                    return klass_result, True

            if instance_result is not "const _sentinel":
                "const cache"[obj, attr] = instance_result, False
                return instance_result, False
            if klass_result is not "const _sentinel":
                "const cache"[obj, attr] = klass_result, "const safe_hasattr"(klass_result, '__get__')
                return "const cache"[obj, attr]

            if obj is klass:
                # for types we check the metaclass too
                for entry in "const _static_getmro"(type(klass)):
                    if "const _shadowed_dict"(type(entry)) is "const _sentinel":
                        try:
                            "const cache"[obj, attr] = entry.__dict__[attr], False
                            return "const cache"[obj, attr]
                        except:  # Assume KeyError
                            pass
            if default is not "const _sentinel":
                "const cache"[obj, attr] = default, False
                return default, False
            raise AttributeError(attr)

    _patch(getattr_static_module.getattr_static, getattr_static)


def optimize_classfilter_filter():
    """
    Replaces 'ClassFilter._filter' and inlines '_access_possible'.
    """
    from jedi.inference.value.klass import ClassFilter
    super_filter = _get_super_meth(ClassFilter, "_filter")
    startswith_meth = str.startswith
    endswith_meth = str.endswith
    append_meth = list.append
    # TODO: inject consts
    def _filter(self, names):
        ret = []
        _is_instance = self._is_instance
        _equals_origin_scope = self._equals_origin_scope
        for name in super_filter(self, names):
            if not _is_instance:
                try:  # Most likely
                    if (expr_stmt := name.get_definition()).type == 'expr_stmt':
                        if (annassign := expr_stmt.children[1]).type == 'annassign':
                            # If there is an =, the variable is obviously also
                            # defined on the class.
                            if 'ClassVar' not in annassign.children[1].get_code() and '=' not in annassign.children:
                                continue
                except:  # Assume AttributeError
                    pass
            if not startswith_meth(v := name.value, "__") or endswith_meth(v, "__") or _equals_origin_scope():
                append_meth(ret, name)
        return ret
    ClassFilter._filter = _filter
    del ClassFilter, _filter


def optimize_get_cached_parent_scope():
    cache = _register_cache()
    from jedi import parser_utils
    from jedi.parser_utils import get_parent_scope

    def get_cached_parent_scope(parso_cache_node, node, include_flows=False):
        if parso_cache_node is None:  # TODO: Find out if not having this breaks anything.
            return get_parent_scope(node, include_flows)
        try:
            return cache[parso_cache_node][node]
        except:  # Assume KeyError
            if parso_cache_node not in cache:
                cache[parso_cache_node] = {}
            ret = cache[parso_cache_node][node] = get_parent_scope(node, include_flows)
            return ret

    assert _is_patched(parser_utils.get_parent_scope), "parser_utils.get_parent_scope was never patched"
    _patch(parser_utils.get_cached_parent_scope, get_cached_parent_scope)
    # XXX Needed because later optimizations require access to the cache closure.
    parser_utils.get_cached_parent_scope = get_cached_parent_scope


# TODO: optimize ClassFilter._access_possible. It depends on cached parser scope


def optimize_leaf_end_pos_descriptor():
    """
    Leaf.end_pos is a descriptor that can be called several thousands times
    per instance. This caches the property via dict.__getitem__.
    """
    from parso.tree import Leaf
    splitlines = str.splitlines
    class Cache(dict):
        def __missing__(self, instance):

            # (split_lines(instance.value))
            string = instance.value
            lines = splitlines(string)
            try:
                if string[-1] is "\n":
                    lines.append("")
            except:  # Always assume IndexError, but skip the lookup.
                lines.append("")

            end_pos_line = instance.line + len(lines) - 1
            # Check for multiline token
            if instance.line == end_pos_line:
                end_pos_column = instance.column + len(lines[-1])
            else:
                end_pos_column = len(lines[-1])
            ret = self[instance] = (end_pos_line, end_pos_column)
            return ret
    Leaf.end_pos_cache = _register_cache(Cache())
    Leaf.end_pos = property(Leaf.end_pos_cache.__getitem__)
    del Leaf, Cache

    # XXX TODO: optimization breaks jedi when fast_parser == True
    import jedi
    jedi.settings.fast_parser = False


def _get_super_meth(cls, methname: str):
    for base in cls.mro()[1:]:
        try:
            return getattr(base, methname)
        except AttributeError:
            continue
    raise Exception(f"Method '{methname}' not on any superclass")


def optimize_check_flows():
    from jedi.inference.filters import ParserTreeFilter
    from jedi.inference.flow_analysis import REACHABLE, UNREACHABLE, reachability_check
    from operator import attrgetter

    @_inject_const(cache=_register_cache(),
                  append=list.append,
                  key=attrgetter("line", "column"),
                #   key=attrgetter("start_pos"),
                  REACHABLE=REACHABLE,
                  UNREACHABLE=UNREACHABLE,
                  reachability_check=reachability_check)
    def _check_flows(self, names):
        try: return "const cache"[names]
        except:  # Assume KeyError, but skip the lookup.
            ret = []
            context = self._node_context
            value_scope = self._parser_scope
            origin_scope = self._origin_scope
            for name in sorted(names, key="const key", reverse=True):
                check = "const reachability_check"(
                    context=context,
                    value_scope=value_scope,
                    node=name,
                    origin_scope=origin_scope)
                if check is not "const UNREACHABLE": "const append"(ret, name)
                if check is "const REACHABLE": break
            "const cache"[names] = ret
            return ret
    _patch(ParserTreeFilter._check_flows, _check_flows)
    del ParserTreeFilter, REACHABLE, UNREACHABLE, reachability_check, attrgetter, _check_flows


def optimize_parsertreefilter_filter():
    """
    Replaces 'ParserTreeFilter._filter' and inlines '_is_name_reachable' and 'get_cached_parent_scope'.
    """
    from jedi.inference.filters import ParserTreeFilter
    from jedi.parser_utils import get_cached_parent_scope, get_parent_scope

    super_filter = _get_super_meth(ParserTreeFilter, "_filter")

    # Cache is taken from the optimized version
    cache = get_cached_parent_scope.__closure__[0].cell_contents
    assert type(cache) is dict, f"parent scope cache was never overridden, cache was {cache}"
    append_meth = list.append

    """

class StubFilter(ParserTreeFilter):
    name_class = StubName

    def _is_name_reachable(self, name):
        if not super()._is_name_reachable(name):
            return False

        # Imports in stub files are only public if they have an "as"
        # export.
        definition = name.get_definition()
        if definition.type in ('import_from', 'import_name'):
            if name.parent.type not in ('import_as_name', 'dotted_as_name'):
                return False
        n = name.value
        # TODO rewrite direct return
        if n.startswith('_') and not (n.startswith('__') and n.endswith('__')):
            return False
        return True
    """
    from jedi.inference.gradual.stub_value import StubFilter
    is_stub_filter = StubFilter.__instancecheck__

    cache2 = _register_cache()
    flow_cache = _register_cache()
    stub_cache = _register_cache()
    # TODO: inject consts
    def _filter(self, names):
        try: cache_node = cache[parso_cache_node := self._parso_cache_node]
        except: cache_node = cache[parso_cache_node] = _register_cache()

        try:    names = cache2[key := (self._until_position, names)]
        except: names = cache2[key] = (super_filter(self, names))

        is_stub = is_stub_filter(self)
        parser_scope = self._parser_scope
        ret = []

        for name in names:
            parent = name.parent
            ptype = parent.type
            if ptype != "trailer":
                base_node = parent if ptype in {"classdef", "funcdef"} else name
                # TODO: Find out if not having this breaks anything.
                # XXX: This doesn't seem to trigger? Remove line?
                # XXX: Seems to trigger on completions inside functions
                # if self._parso_cache_node is None:
                #     scope = get_parent_scope(base_node, False)
                # else:
                try:
                    scope = cache_node[base_node]
                except:  # Assume KeyError
                    scope = cache_node[base_node] = get_parent_scope(base_node, False)

                if scope is parser_scope:
                    # StubFilter-specific. Removes imports in .pyi
                    if is_stub:
                        try:
                            definition = stub_cache[name]
                        except:
                            definition = stub_cache[name] = name.get_definition()
                        definition = name.get_definition()
                        if definition.type in {'import_from', 'import_name'}:
                            if ptype not in {'import_as_name', 'dotted_as_name'}:
                                continue
                        n = name.value
                        if n == "ellipsis":
                            continue  # Not sure why builtins even include this

                        if n[0] is "_":
                            try:
                                if n[1]  is not "_" or n[-1] is not "_" is not n[-2]:
                                    continue
                            except:
                                continue
                            # if n.startswith('_') and not (n.startswith('__') and n.endswith('__')):
                            #     continue
                    append_meth(ret, name)
        try:
            ret = flow_cache[t := tuple(ret)]
            # p.disable()
            return ret
        except:
            ret = flow_cache[t] = self._check_flows(t)
            # p.disable()
            return ret

        # # NOTE: Don't inline _check_flows - SelfAttributeFilter overrides it.
        # return self._check_flows(tuple(ret))

    ParserTreeFilter._filter = _filter
    del ParserTreeFilter, StubFilter, _filter


def optimize_selfattributefilter_filter():
    """
    Replaces 'SelfAttributeFilter._filter' and inlines '_filter_self_names'.
    """
    from jedi.inference.value.instance import SelfAttributeFilter
    from operator import attrgetter

    @_inject_const(cache=_register_cache(),  # XXX: Cache for start and end descriptors
                   append=list.append,
                   get_ends=attrgetter("start_pos", "end_pos"),
                   cache2=_register_cache())
    def _filter(self, names):
        try: start, end = "const cache"[(scope := self._parser_scope)]
        except: start, end = "const cache"[scope] = "const get_ends"(scope)
        ret = []
        for name in names:
            if name in "const cache2":
                if "const cache2"[name]:
                    "const append"(ret, name)
            else:
                if start < (name.line, name.column) < end:
                    if (trailer := name.parent).type == 'trailer':
                        children = trailer.parent.children
                        if len(children) is 2 and trailer.children[0] is '.':
                            if name.is_definition() and self._access_possible(name):
                                if self._is_in_right_scope(children[0], name):
                                    "const cache2"[name] = True
                                    "const append"(ret, name)
                                    continue
                "const cache2"[name] = False
        return ret
    _patch(SelfAttributeFilter._filter, _filter)


def optimize_split_lines():
    @_inject_const(splitlines=str.splitlines, append=list.append)
    def split_lines(string: str, keepends: bool = False) -> list[str]:
        lines = "const splitlines"(string, keepends)
        try:
            if string[-1] is "\n":
                "const append"(lines, "")
        except:  # Assume IndexError, but skip the lookup
            "const append"(lines, "")
        return lines

    from parso import utils
    _patch(utils.split_lines, split_lines)
    del split_lines, utils


def optimize_fuzzy_match():
    """
    Optimizes '_fuzzy_match' in jedi.api.helpers.
    """
    @_inject_const(index_meth=str.index)
    def _fuzzy_match(string: str, like_name: str):
        index = -1
        try:
            for char in like_name:
                index = "const index_meth"(string, char, index + 1)
                continue
        except:  # Expect IndexError but don't bother looking up the name.
            return False
        return True
    from jedi.api import helpers
    _patch(helpers._fuzzy_match, _fuzzy_match)


def optimize_filter_names():
    from jedi.api.classes import Completion
    from jedi.api.helpers import _fuzzy_match
    from jedi.api import completion
    class NewCompletion(Completion):
        __init__ = object.__init__
        _is_fuzzy = False

    class FuzzyCompletion(Completion):
        __init__ = object.__init__
        _is_fuzzy = True
        complete = property(None.__init__)

    @_inject_const(len=len,
                  fuzzy_match=_fuzzy_match,
                  FuzzyCompletion=FuzzyCompletion,
                  NewCompletion=NewCompletion,
                  startswith=str.startswith,
                  append=list.append,
                  lower=str.lower,
                  strip=str.strip,
                  add=set.add)
    def filter_names(inference_state, completion_names, stack, like_name, fuzzy, cached_name):
        comp_dct = set()
        like_name_len = "const len"(like_name)
        if fuzzy:
            match_func = "const fuzzy_match"
            completion_base = "const FuzzyCompletion"
            do_complete = None.__init__         # Dummy
        else:
            match_func = "const startswith"
            completion_base = "const NewCompletion"
            if settings.add_bracket_after_function:
                def do_complete():
                    if new.type == "function":
                        return string_name[like_name_len:] + "("
                    return string_name[like_name_len:]
            else:
                def do_complete():
                    return string_name[like_name_len:]

        class completion(completion_base):
            _inference_state = inference_state  # InferenceState object is global.
            _like_name_length = like_name_len   # Static
            _cached_name = cached_name          # Static
            _stack = stack                      # Static

        if settings.case_insensitive_completion:
            case = "const lower"
            like_name = case(like_name)
        else:
            case = "const strip"  # Dummy
        ret = []
        for name in completion_names:
            string_name = name.string_name
            if match_func(case(string_name), like_name):
                new = completion()
                new._name = name
                # new._same_name_completions = []  # XXX Not even used?
                k = (string_name, do_complete())  # key
                if k not in comp_dct:
                    "const add"(comp_dct, k)
                    tree_name = name.tree_name
                    if tree_name is not None:
                        definition = tree_name.get_definition()
                        if definition is not None and definition.type == 'del_stmt':
                            continue
                    "const append"(ret, new)
        return ret
    # return filter_names
    _patch(completion.filter_names, filter_names)
    del Completion, completion, _fuzzy_match, filter_names


def optimize_get_parent_scope():
    from parso.python import tree

    @_inject_const(
        is_flow=tree.Flow.__instancecheck__)
    def get_parent_scope(node, include_flows=False):
        scope = node.parent

        if scope is not None:

            tmp = node
            try:
                while True:tmp = tmp.children[0]
            except:
                node_start_pos = tmp.line, tmp.column

            node_parent = node.parent
            parent_type = node_parent.type

            if (cont := not (parent_type == 'param' and node_parent.name == node)):
                if (cont := not (parent_type == 'tfpdef' and node_parent.children[0] == node)):
                    pass

            while True:
                if ((t := scope.type) == 'comp_for' and scope.children[1].type != 'sync_comp_for') or \
                        t in {'file_input', 'classdef', 'funcdef', 'lambdef', 'sync_comp_for'}:
                    if scope.type in {'classdef', 'funcdef', 'lambdef'}:

                        for child in scope.children:
                            try:
                                if child.value is ":": break
                            except: continue
                        if (child.line, child.column) >= node_start_pos:
                            if cont:
                                scope = scope.parent
                                continue
                    return scope
                elif include_flows and "const is_flow"(scope):
                    if t == 'if_stmt':
                        for n in scope.get_test_nodes():
                            tmp = n
                            try:
                                while True: tmp = tmp.children[0]
                            except:start = tmp.line, tmp.column
                            if start <= node_start_pos:
                                tmp = n
                                try:
                                    while True: tmp = tmp.children[-1]
                                except:end = tmp.end_pos
                                if node_start_pos < end:
                                    break
                        else:
                            return scope
                scope = scope.parent
        return None  # It's a module already.

    from jedi import parser_utils
    _patch(parser_utils.get_parent_scope, get_parent_scope)


def optimize_importfrom_get_defined_names():
    from parso.python.tree import ImportFrom

    def get_defined_names(self, _):
        try:
            
            if (value := (last := (children := self.children)[-1]).value) in {"*", ")"}:
                if value is ")":
                    last = children[-2]
                else:
                    return ()
        except:
            pass

        if last.type == 'import_as_names':
            as_names = last.children[::2]
        else:
            as_names = [last]
        ret = []
        for as_name in as_names:
            if as_name.type == 'name':
                ret.append(as_name)
            else:
                ret.extend(x or y for x, y in as_name.children[::2])
        return ret


def optimize_memoize_method():
    @_inject_const(cache=_register_cache(), tuple=tuple, dict_items=dict.items)
    def memoize_method(method):
        from types import MethodType
        assert isinstance(method, MethodType)

        "const meth_cache"[method.__func__] = _register_cache()

        def wrapper(self, *args, **kwargs):
            try:    d = "const cache"[key := (self, "const meth")]
            except: d = "const cache"[key] = {}

            key = (args, "const tuple"("const dict_items"(kwargs)))
            try:
                return d[key]
            except:
                d[key] = result = method(self, *args, **kwargs)
                return result
        return wrapper


def optimize_valuewrapperbase_getattr():
    """
    Replaces _ValueWrapperBase.__getattr__.

    __getattr__ works like dict's __missing__. When an attribute error is
    thrown, use this opportunity to set a cached value.
    """
    @_inject_const(getattr=getattr, setattr=setattr)
    def __getattr__(self, name):
        ret = "const getattr"(self._wrapped_value, name)
        "const setattr"(self, name, ret)
        return ret
    from jedi.inference.base_value import _ValueWrapperBase
    _patch(_ValueWrapperBase.__getattr__, __getattr__)


def optimize_lazyvaluewrapper_wrapped_value():
    def _wrapped_value(self):
        if "_cached_value" in (d := self.__dict__):
            return d["_cached_value"]
        result = d["_cached_value"] = self._get_wrapped_value()
        return result

    from jedi.inference.base_value import LazyValueWrapper
    LazyValueWrapper._wrapped_value = property(_wrapped_value)
