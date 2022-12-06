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


def defer_patch(orig, replace):
    # Evil, but the only reliable way.
    def func(*args, **kw):
        from sys import _getframe
        f = _getframe(0)
        new = f.f_code.co_consts[-1]
        assert (f_back := f.f_back)
        f_back.f_globals[f.f_code.co_name] = new
        return new(*args, **kw)

    orig.__code__ = func.__code__.replace(
        co_name=orig.__code__.co_name,
        co_consts=func.__code__.co_consts + (replace,))


def _clear_caches(_caches=[], _clear=dict.clear) -> None:
    """
    Internal.
    Evict all caches passed to, or obtained by, _register_cache().
    """
    for cache in _caches:
        _clear(cache)
    return None


def _register_cache(cache: dict = None) -> dict:
    if cache is None:
        cache = {}
    assert isinstance(cache, dict), f"Expected dict or None, got {type(cache)}"
    _clear_caches.__defaults__[0].append(cache)
    return cache


def setup2():
    import jedi
    import parso
    assert jedi.__version__ == _version_compat_jedi
    assert parso.__version__ == _version_compat_parso

    optimize_platform_system()
    optimize_name_is_definition()
    optimize_get_used_names()
    optimize_valuewrapperbase_name()
    optimize_classfilter_filter()       # Call after optimize_parsertreefilter_filter
    optimize_get_definition_names()
    # optimize_selfattributefilter_filter()
    optimize_split_lines()
    optimize_fuzzy_match()
    optimize_filter_names()
    optimize_valuewrapperbase_getattr()
    optimize_lazyvaluewrapper_wrapped_value()


class Interpreter(Script):
    __slots__ = ()
    _inference_state = InferenceState(Project(Path("~")), InterpreterEnvironment(), None)
    _inference_state.allow_descriptor_getattr = True

    def __new__(cls, *args, **kw):
        self = super().__new__(cls)
        from jedi.inference.imports import import_module_by_names
        InferenceState.builtins_module, = import_module_by_names(self._inference_state, ("builtins",), sys_path=(), prefer_stubs=False)
        Interpreter.__new__ = lambda *args, **kw: self
        return self

    def __init__(self, code: str, _=[]):
        # Workaround for jedi's recursion limit design.
        self._inference_state.inferred_element_counts.clear()
        self._module_node = self._inference_state.grammar.parse(code=code, diff_cache=settings.fast_parser, cache_path=settings.cache_directory)
        self._code_lines = lines = code.splitlines(True)
        self._code = code
        try:
            if code[-1] is "\n":
                lines.append("")
        except:
            lines.append("")

    def _get_module_context(self):
        tree_module_value = ModuleValue(self._inference_state, self._module_node, file_io=None, string_names=('__main__',), code_lines=self._code_lines)
        return MixedModuleContext(tree_module_value, ())

    def complete(self, line, column, *, fuzzy=False):
        return Completion(self._inference_state, self._get_module_context(), self._code_lines, (line, column), self.get_signatures, fuzzy=fuzzy).complete()

    __repr__ = object.__repr__


def optimize_platform_system():
    """Replace platform.system with sys.platform"""
    import sys
    import platform
    platform._system = platform.system  # Backup
    platform.system = sys.platform.replace("win32", "windows").title


def optimize_name_is_definition():
    from parso.python.tree import Name, _GET_DEFINITION_TYPES

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
            if node.type in _GET_DEFINITION_TYPES:
                if self in node.get_defined_names(include_setitem):
                    return node
                break
            node = node.parent
        return None
    Name.is_definition = is_definition


def optimize_get_used_names():
    """Replaces parso.python.tree.Module.get_used_names"""

    _hasattr = hasattr
    _extend = list.extend
    _append = list.append
    _pop = list.pop

    def get_used_names(self):
        if used_names := self._used_names:
            return used_names
        pool = [self]
        self._used_names = used_names = {}
        while pool:
            node = _pop(pool)
            if _hasattr(node, "children"):
                _extend(pool, node.children)
            elif node.type == "name":
                if (value := node.value) in used_names:
                    _append(used_names[value], node)
                else:
                    used_names[value] = [node]
        return used_names

    from parso.python.tree import Module, UsedNamesMapping
    Module.get_used_names = get_used_names


def optimize_valuewrapperbase_name():
    from jedi.inference.names import ValueName
    from jedi.inference.compiled import CompiledValueName
    from jedi.inference.utils import UncaughtAttributeError
    from jedi.inference import base_value

    def name(self):
        try:
            if name := self._wrapped_value.name.tree_name:
                return ValueName(self, name)
            return CompiledValueName(self, name)
        except AttributeError as e:
            raise UncaughtAttributeError(e) from e
    base_value._ValueWrapperBase.name = property(name)


def optimize_get_definition_names():
    """Optimizes _get_definition_names in 'jedi/inference/filters.py'"""
    cache = _register_cache()
    def _get_definition_names(parso_cache_node, used_names, name_key):
        try:
            return cache[parso_cache_node][name_key]
        except:  # Assume KeyError, but skip the error lookup
            try:
                ret = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
            except:  # Unlikely. Assume KeyError, but skip the error lookup
                ret = ()
            if parso_cache_node is not None:
                try:
                    cache[parso_cache_node][name_key] = ret
                except:  # Unlikely. Assume KeyError, but skip the error lookup
                    cache[parso_cache_node] = {name_key: ret}
            return ret

    from jedi.inference import filters
    filters._get_definition_names = _get_definition_names


def optimize_classfilter_filter():
    """Replaces 'ClassFilter._filter' and inlines '_access_possible'."""
    from jedi.inference.value.klass import ClassFilter
    super_filter = _get_super_meth(ClassFilter, "_filter")
    startswith = str.startswith
    endswith = str.endswith
    append = list.append

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
            if not startswith(v := name.value, "__") or endswith(v, "__") or _equals_origin_scope():
                append(ret, name)
        return ret
    ClassFilter._filter = _filter


def _get_super_meth(cls, methname: str):
    for base in cls.mro()[1:]:
        try:
            return getattr(base, methname)
        except AttributeError:
            continue
    raise Exception(f"Method '{methname}' not on any superclass")


def optimize_selfattributefilter_filter():
    """
    Replaces 'SelfAttributeFilter._filter' and inlines '_filter_self_names'.
    """
    from jedi.inference.value.instance import SelfAttributeFilter
    from operator import attrgetter
    cache = _register_cache()  # XXX: Cache for start and end descriptors
    append = list.append
    get_ends = attrgetter("start_pos", "end_pos")
    cache2 = _register_cache()
    def _filter(self, names):
        try: start, end = cache[(scope := self._parser_scope)]
        except: start, end = cache[scope] = get_ends(scope)
        ret = []
        for name in names:
            if name in cache2:
                if cache2[name]:
                    append(ret, name)
            else:
                if start < (name.line, name.column) < end:
                    if (trailer := name.parent).type == 'trailer':
                        children = trailer.parent.children
                        if len(children) is 2 and trailer.children[0] is '.':
                            if name.is_definition() and self._access_possible(name):
                                if self._is_in_right_scope(children[0], name):
                                    cache2[name] = True
                                    append(ret, name)
                                    continue
                cache2[name] = False
        return ret
    SelfAttributeFilter._filter = _filter


def optimize_split_lines():
    splitlines = str.splitlines
    append = list.append

    def split_lines(string: str, keepends: bool = False) -> list[str]:
        lines = splitlines(string, keepends)
        try:
            if string[-1] is "\n":
                append(lines, "")
        except:  # Assume IndexError, but skip the lookup
            append(lines, "")
        return lines

    from parso import utils
    utils.split_lines = split_lines


def optimize_fuzzy_match():
    """Optimizes '_fuzzy_match' in jedi.api.helpers."""
    _index = str.index
    def _fuzzy_match(string: str, like_name: str):
        index = -1
        try:
            for char in like_name:
                index = _index(string, char, index + 1)
                continue
        except:  # Expect IndexError but don't bother looking up the name.
            return False
        return True
    from jedi.api import helpers
    helpers._fuzzy_match = _fuzzy_match


def optimize_filter_names():
    from jedi.api.classes import Completion
    from jedi.api.helpers import _fuzzy_match
    from jedi.api import completion
    class NewCompletion(Completion):
        __init__ = None.__init__
        _is_fuzzy = False

    class FuzzyCompletion(Completion):
        __init__ = None.__init__
        _is_fuzzy = True
        complete = property(None.__init__)

    _len = len
    fuzzy_match = _fuzzy_match
    startswith = str.startswith
    append = list.append
    lower = str.lower
    strip = str.__str__

    def filter_names(inference_state, completion_names, stack, like_name, fuzzy, cached_name):
        like_name_len = _len(like_name)
        if fuzzy:
            match_func = fuzzy_match
            completion_base = FuzzyCompletion
            do_complete = None.__init__         # Dummy
        else:
            match_func = startswith
            completion_base = NewCompletion
            if settings.add_bracket_after_function:
                def do_complete():
                    if new.type == "function":
                        return n[like_name_len:] + "("
                    return n[like_name_len:]
            else:
                def do_complete():
                    return n[like_name_len:]

        class completion(completion_base):
            _inference_state = inference_state  # InferenceState object is global.
            _like_name_length = like_name_len   # Static
            _cached_name = cached_name          # Static
            _stack = stack                      # Static

        if settings.case_insensitive_completion:
            case = lower
            like_name = case(like_name)
        else:
            case = strip  # Dummy
        ret = []
        dct = {}
        for e in completion_names:
            n_ = e.string_name
            n = case(n_)
            if n[:like_name_len] == like_name:
                # Store unmodified so names like Cache and cache aren't merged.
                if (k := (n_, n_[like_name_len:])) not in dct:
                    dct[k] = None
                    if ((tn := e.tree_name) is None) or (getattr(tn.get_definition(), "type", None) != "del_stmt"):
                        new = completion()
                        new._name = e
                        append(ret, new)
        return ret
    completion.filter_names = filter_names


def optimize_valuewrapperbase_getattr():
    """Replaces _ValueWrapperBase.__getattr__.
    __getattr__ works like dict's __missing__. When an attribute error is
    thrown, use this opportunity to set a cached value.
    """
    _getattr = getattr
    _setattr = setattr
    def __getattr__(self, name):
        ret = _getattr(self._wrapped_value, name)
        _setattr(self, name, ret)
        return ret
    from jedi.inference.base_value import _ValueWrapperBase
    _ValueWrapperBase.__getattr__ = __getattr__


def optimize_lazyvaluewrapper_wrapped_value():
    def _wrapped_value(self):
        if "_cached_value" in (d := self.__dict__):
            return d["_cached_value"]
        result = d["_cached_value"] = self._get_wrapped_value()
        return result

    from jedi.inference.base_value import LazyValueWrapper
    LazyValueWrapper._wrapped_value = property(_wrapped_value)
