# This adds fixes for jedi to outright work.

from textension.utils import _patch_function, _forwarder, inline
from .tools import _descriptor_overrides, _value_overrides, _virtual_overrides

import bpy


def apply():
    # Needs to run before any optimization that modifies pickled objects.
    patch_load_from_file_system()

    _apply_optimizations()
    _apply_patches()


def _apply_patches():
    from . import modules
    modules._blf.apply()
    modules._gpu.apply()
    modules._bpy.apply()
    modules._bpy_types.apply()
    modules._mathutils.apply()

    # Remove the bulk of ``__getattr__`` calls.
    patch_NameWrapper_getattr()
    patch_ExactValue()
    patch_ValueWrapperBase()

    patch_Completion_complete_inherited()
    patch_fakelist_array_type()
    patch_import_resolutions()
    patch_get_builtin_module_names()
    patch_compiledvalue()  # XXX: Not having this means tuple[X] isn't subscriptable.
    patch_is_pytest_func()
    patch_paths_from_list_modifications()

    patch_create_cached_compiled_value()
    patch_various_redirects()
    patch_SequenceLiteralValue()
    patch_complete_dict()
    patch_get_user_context()
    patch_Importer()
    patch_get_importer_names()
    patch_getattr_stack_overflows()
    patch_cache_signatures()
    patch_AbstractContext_check_for_additional_knowledge()
    patch_DirectAccess_key_errors()
    patch_parse_function_doc()
    patch_SignatureMixin_to_string()
    patch_CompiledValueFilter_get_cached_name()
    patch_BaseName_get_docstring()
    patch_canon_typeshed_compatibility()
    patch_Script_get_signatures()
    patch_HelperValueMixin_is_same_class()
    patch_Completion_complete_trailer()
    patch_BaseName_get_signatures()

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
    from jedi.inference.gradual.typing import TypedDict
    from jedi.inference.value import TreeInstance
    from .common import state, state_cache, state_cache_kw, Values

    @state_cache
    def py__bases__(self: GenericClass):
        ret = []
        for base in self._wrapped_value.py__bases__():
            ret += _LazyGenericBaseClass(self, base, self._generics_manager),
        return ret

    GenericClass.py__bases__ = py__bases__


    @state_cache_kw
    def py__call__(self: GenericClass, arguments):
        if self.is_typeddict():
            return Values((TypedDict(self),))
        return Values((TreeInstance(state, self.parent_context, self, arguments),))

    GenericClass.py__call__ = py__call__


# Removes calls to __getattr__ for dynamic forwarding. This is an effort to
# eliminate most stack overflows during debugging.
def patch_various_redirects():
    from textension.utils import _soft_forwarder, soft_property
    from jedi.inference.value.iterable import SequenceLiteralValue

    SequenceLiteralValue.parent_context = _forwarder("_wrapped_value.parent_context")
    SequenceLiteralValue.py__class__    = _forwarder("_wrapped_value.py__class__")
    SequenceLiteralValue._arguments     = _forwarder("_wrapped_value._arguments")
    SequenceLiteralValue.is_module      = _forwarder("_wrapped_value.is_module")
    SequenceLiteralValue.is_stub        = _forwarder("_wrapped_value.is_stub")
    SequenceLiteralValue.is_instance    = _forwarder("_wrapped_value.is_instance")
    SequenceLiteralValue.__iter__       = _forwarder("_wrapped_value.__iter__")
    SequenceLiteralValue.array_type     = _soft_forwarder("_wrapped_value.array_type")

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

    @soft_property
    def _class_value(self):
        raise AttributeError
    GenericClass._class_value = _class_value

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


# Fixes jedi erroneously yielding wrong inherited methods.
# Also fixes no ancestor found because of end marker
def patch_Completion_complete_inherited():
    from jedi.api.completion import Completion
    from parso.python.tree import search_ancestor
    from itertools import islice

    def _complete_inherited(self: Completion, is_function=True):

        ret = []
        leaf = self._module_node.get_leaf_for_position(self._position, include_prefixes=True)

        # Getting previous leaf could return None.
        if leaf.type != "endmarker" or (leaf := leaf.get_previous_leaf()):
            cls = search_ancestor(leaf, 'classdef')

            if cls and cls.start_pos[1] < leaf.start_pos[1]:
                # Complete the methods that are defined in the super classes.
                value = self._module_context.create_value(cls)

                filters = value.get_filters(is_instance=True)
                # The first dict is the dictionary of class itself.
                for filter in islice(filters, 1, None):
                    for name in filter.values():
                        if (name.api_type == "function") is is_function:
                            ret += name,
        return ret

    Completion._complete_inherited = _complete_inherited


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
# gpu, mathutils, etc.
def patch_import_resolutions():

    from jedi.inference.value.namespace import ImplicitNamespaceValue
    from jedi.inference.imports import import_module, _load_builtin_module, \
        ImplicitNSInfo, _load_python_module
    import sys

    # ``import_module`` undecorated.
    import_module = import_module.__closure__[0].cell_contents["import_module"]
    import_module = import_module.__closure__[0].cell_contents
    import_module = import_module.__closure__[0].cell_contents

    from .common import Values, NO_VALUES

    @inline
    def is_implicit_ns_info(obj) -> bool:
        return ImplicitNSInfo.__instancecheck__

    def fixed_import_module(inference_state, import_names, parent_module_value, sys_path):
        # XXX: We don't allow auto imports.
        # if import_names[0] in settings.auto_import_modules:
        #     if module := _load_builtin_module(inference_state, import_names, sys_path):
        #         return Values((module,))
        #     return NO_VALUES

        module_name = '.'.join(import_names)
        if parent_module_value is None:
            file_io_or_ns, is_pkg = inference_state.compiled_subprocess.get_module_info(string=import_names[-1], full_name=module_name, sys_path=sys_path, is_global_search=True)

        else:
            if paths := parent_module_value.py__path__():
                file_io_or_ns, is_pkg = inference_state.compiled_subprocess.get_module_info(string=import_names[-1], path=paths, full_name=module_name, is_global_search=False)

            else:
                # No paths means the module is file-less. We need to nudge
                # things along. Mathutils submodules are like this.
                is_pkg = None
                file_io_or_ns = None
                if module_name in sys.modules:
                    is_pkg = False

        if is_pkg is not None:
            if is_implicit_ns_info(file_io_or_ns):
                module = ImplicitNamespaceValue(inference_state, string_names=tuple(file_io_or_ns.name.split('.')), paths=file_io_or_ns.paths)
            elif file_io_or_ns:
                module = _load_python_module(inference_state, file_io_or_ns, import_names=import_names, is_package=is_pkg)
            else:
                module = _load_builtin_module(inference_state, import_names, sys_path)
            return Values((module,))
        return NO_VALUES

    _patch_function(import_module, fixed_import_module)


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
    from jedi.inference.imports import Importer, import_module_by_names
    from .common import Importer_redirects, state, Values, NO_VALUES
    from jedi.inference.value.namespace import ImplicitNamespaceValue
    import os

    Importer._inference_state = state
    Importer._fixed_sys_path = None
    Importer._infer_possible = True

    # Remove the property. We write the value just once in __init__.
    Importer._str_import_path = None

    # Import redirection for actual modules.
    def follow(self: Importer):
        import_path = self._str_import_path
        if module := Importer_redirects.get(".".join(import_path)):
            return Values((module,))
        
        if not self.import_path:
            if self._fixed_sys_path:
                import_path = (os.path.basename(self._fixed_sys_path[0]),)
                ns = ImplicitNamespaceValue(state, import_path,self._fixed_sys_path)
                return Values((ns,))
            return NO_VALUES
        if not self._infer_possible:
            return NO_VALUES

        if from_cache := state.stub_module_cache.get(import_path):
            return Values((from_cache,))
        if from_cache := state.module_cache.get(import_path):
            return from_cache

        # Use the real sys path, not dynamically modified.
        sys_path = state.project._get_sys_path(add_init_paths=True)

        return import_module_by_names(
            state, self.import_path, sys_path, self._module_context)

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
        self.import_path = self._str_import_path = tuple(map(str, self.import_path))

    __init__ = _patch_function(Importer.__init__, __init__)

    # The ONLY place jedi passes a list of NAMES to the importer, and for some
    # reason thought it would be a good idea to keep it like that (it's not).
    # This also removes the try/except clause for a more performant version.
    from jedi.inference.imports import _prepare_infer_import, search_ancestor
    from parso.python.tree import ImportFrom

    def prepare_infer_import(module_context, tree_name):
        import_node = search_ancestor(tree_name, 'import_name', 'import_from')
        import_path = import_node.get_path_for_name(tree_name)
        from_name = None

        if import_node.__class__ is ImportFrom:
            from_names = import_node.get_from_names()
            if len(from_names) + 1 == len(import_path):
                from_name = import_path[-1]
                import_path = from_names

        # Assumes Leaf.__str__ unbound getter optimization.
        import_path = tuple(map(str, import_path))
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
                tmp += name,
        return builtin | set(tmp)
    
    _patch_function(functions.get_builtin_module_names, get_builtin_module_names)


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
    from .modules._bpy_types import get_rna_value, is_bpy_struct
    from .common import CompiledModule_redirects, state, state_cache
    from .tools import get_handle

    bpy_types = (RNAMeta, StructRNA, StructMetaPropGroup, bpy.props._PropertyDeferred)

    @state_cache
    def create_cached_compiled_value_p(state, handle, context):
        obj = handle.access._obj

        if context:
            if isinstance(obj, bpy_types) or (isinstance(obj, type) and issubclass(obj, bpy_types) and is_bpy_struct(getattr(obj, "bl_rna", None))):
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

        try:
            if obj in _virtual_overrides:
                return _virtual_overrides[obj]((obj, context._value))
        except TypeError:
            pass

        try:
            if obj in _descriptor_overrides:
                handle = get_handle(_descriptor_overrides[obj])
                return create_cached_compiled_value(state, handle, context)
        except TypeError:
            pass

        # Some objects are unhashable, but they are too rare and not
        # worth the extra overhead of using isinstance for.
        try:
            if obj in _value_overrides:
                return _value_overrides[obj](obj, value)
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
    from jedi.inference.imports import SubModuleName, Importer
    from jedi.api.completion import Completion, _gather_nodes
    from itertools import repeat, compress
    import sys
    from importlib.util import find_spec
    from .common import filter_operators, is_namenode

    startswith = str.startswith
    modules = sys.modules.keys()

    def get_importer_names(self: Completion, names, level=0, only_modules=True):
        
        string_names = [n.value for n in names]
        i = Importer(self._inference_state, string_names, self._module_context, level)

        for module in i.follow():
            context = module.as_context()
            # This fix applies only when there's at least 1 import name and 1 dot
            # operator. In this context, dots, comma and parentheses may appear.
            for op in names and filter_operators(_gather_nodes(self.stack)):
                if op.value is ".":
                    # Compose the import statement and look for it in sys.modules.
                    comp = ".".join(string_names)

                    if "." in comp:
                        if module := sys.modules.get(comp):
                            spec = getattr(module, "__spec__", None)
                        else:
                            spec = find_spec(comp)

                        # If a module or spec exists, run stock behavior.
                        if spec:
                            break

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
                            ret += name,
                    if ret:
                        
                        return [SubModuleName(context, n) for n in ret]
                        # return [ImportName(self._module_context, n) for n in ret]

        # TODO: Can we use this? (commented)
        # return i.completion_names(self._inference_state, only_modules=only_modules)
        return get_importer_names(self, names, level=level, only_modules=only_modules)

    get_importer_names = _patch_function(Completion._get_importer_names, get_importer_names)


def patch_ExactValue():
    from jedi.inference.compiled import ExactValue
    from textension.utils import truthy_noargs, falsy_noargs
    from .common import state

    # We don't want the indirection that tests against a list of names.
    # Instead we forward them to known data paths.
    del ExactValue.__getattribute__

    # See ExactValue.__getattribute__.
    ExactValue.access_handle     = _forwarder("_compiled_value.access_handle")
    ExactValue.execute_operation = _forwarder("_compiled_value.execute_operation")
    ExactValue.get_safe_value    = _forwarder("_compiled_value.get_safe_value")
    ExactValue.py__bool__        = _forwarder("_compiled_value.py__bool__")
    ExactValue.negate            = _forwarder("_compiled_value.negate")
    ExactValue.is_compiled       = truthy_noargs

    # These end up going to LazyValueWrapper._wrapped_value stub value.
    ExactValue.is_bound_method   = _forwarder("_wrapped_value.is_bound_method")
    ExactValue.parent_context    = _forwarder("_wrapped_value.parent_context")
    ExactValue._arguments        = _forwarder("_wrapped_value._arguments")
    ExactValue.is_stub           = falsy_noargs
    ExactValue.get_filters       = _forwarder("_wrapped_value.get_filters")

    def __init__(self: ExactValue, compiled_value):
        self._compiled_value = compiled_value

    ExactValue.__init__ = __init__
    ExactValue.inference_state = state


def patch_ValueWrapperBase():
    from jedi.inference.gradual.typing import TypedDict
    from jedi.inference.base_value import _ValueWrapperBase
    from jedi.plugins.stdlib import EnumInstance
    from .common import state

    _ValueWrapperBase.inference_state = state
    _ValueWrapperBase.__iter__        = _forwarder("_wrapped_value.__iter__")

    _ValueWrapperBase.is_bound_method = _forwarder("_wrapped_value.is_bound_method")
    _ValueWrapperBase.is_instance     = _forwarder("_wrapped_value.is_instance")
    _ValueWrapperBase.tree_node       = _forwarder("_wrapped_value.tree_node")

    # Forwarding makes attributes immutable, clear the descriptor on these.
    TypedDict.tree_node = None
    EnumInstance.tree_node = None

    _ValueWrapperBase.py__doc__       = _forwarder("_wrapped_value.py__doc__")

    _ValueWrapperBase.get_default_param_context = _forwarder("_wrapped_value.get_default_param_context")


def patch_getattr_stack_overflows():
    from jedi.plugins.stdlib import SuperInstance
    from textension.utils import _soft_attribute_error

    SuperInstance.inference_state = _soft_attribute_error
    SuperInstance._instance = _soft_attribute_error


# ``cache_signatures`` uses an unmanaged cache and causes inferred values to
# be retained indefinitely. This just skips the cache.
def patch_cache_signatures():
    from jedi.api.helpers import infer, cache_signatures

    def _cache_signatures(inference_state, context, bracket_leaf, code_lines, user_pos):
        return infer(inference_state, context, bracket_leaf.get_previous_leaf())

    _patch_function(cache_signatures, _cache_signatures)


# Fixes ``get_parent_scope`` returning None when ``flow_scope`` is a module.
def patch_AbstractContext_check_for_additional_knowledge():
    from jedi.inference.context import AbstractContext
    from jedi.inference.finder import check_flow_information
    from .common import is_namenode, NO_VALUES, walk_scopes

    def _check_for_additional_knowledge(self, name_or_str, context, position):
        context = context or self

        if is_namenode(name_or_str) and not context.is_instance():
            flow  = name_or_str
            module_node = context.get_root_context().tree_node
            base_nodes  = {context.tree_node, module_node}

            if not any(b.type in {"comp_for", "sync_comp_for"} for b in base_nodes):
                for flow in walk_scopes(flow, False):
                    if n := check_flow_information(context, flow, name_or_str, position):
                        return n
                    elif flow in base_nodes:
                        break
        return NO_VALUES

    AbstractContext._check_for_additional_knowledge = _check_for_additional_knowledge


# Handles are cleared to stop memory leaks. Jedi reports KeyError on handles,
# which is weird considering jedi itself chooses the extra indirection. This
# patch simply bypasses the whole indirection.
def patch_DirectAccess_key_errors():
    from jedi.inference.compiled.subprocess import AccessHandle

    # XXX: Not needed?
    # def py__path__(self: AccessHandle):
    #     return self.access.py__path__()
    # AccessHandle.py__path__ = py__path__

    AccessHandle.__getattr__ = _forwarder("access.__getattribute__")


# Because it's broken. ``func(arg=-1)`` becomes ``func(arg=_1)``.
def patch_parse_function_doc():
    from jedi.inference.compiled.value import _parse_function_doc, docstr_defaults
    import re
    ret_match = re.compile(r'(,\n|[^\n-])+').match

    def parse_function_doc(doc: str):
        param_str = ""
        ret = ""
        end = -1

        if start := doc.find("(") + 1:
            end = doc.find(")", start)
            if end is not -1:
                param_str = doc[start:end]
                end = doc.find("-> ", end)
                if end is not -1:
                    ret_str = ret_match(doc, end + 3).group(0).strip()
                    # New object -> object()
                    ret_str = re.sub(r'[nN]ew (.*)', r'\1()', ret_str)
                    ret = docstr_defaults.get(ret_str, ret_str)
        return param_str, ret

    _patch_function(_parse_function_doc, parse_function_doc)


# Patches ``to_string`` to give a more sane signature.
# No one cares about positional/keyword delimiter noise in parameters.
def patch_SignatureMixin_to_string():
    from jedi.inference.signature import _SignatureMixin
    from inspect import Parameter

    POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
    VAR_POSITIONAL = Parameter.VAR_POSITIONAL
    KEYWORD_ONLY = Parameter.KEYWORD_ONLY

    def to_string(self: _SignatureMixin):
        string_name = self.name.string_name

        if self.name.api_type == "module":
            return f"Module '{string_name}'"

        param_strings = []
        is_kw_only = False
        is_positional = False

        for n in self.get_param_names(resolve_stars=True):
            kind = n.get_kind()
            is_positional |= kind == POSITIONAL_ONLY

            if is_positional and kind != POSITIONAL_ONLY:
                is_positional = False

            if kind == VAR_POSITIONAL:
                is_kw_only = True

            elif kind == KEYWORD_ONLY and not is_kw_only:
                is_kw_only = True

            param_strings += n.to_string(),

        s = f"{string_name}({', '.join(param_strings)})"
        if annotation := self.annotation_string:
            s += " -> " + annotation
        return s

    _SignatureMixin.to_string = to_string


# Jedi doesn't infer property objects, because it doesn't know how, I guess?
# Well this kind of does, at least if the decoratee is a FunctionType.
def patch_CompiledValueFilter_get_cached_name():
    from jedi.inference.compiled.getattr_static import getattr_static
    from jedi.inference.compiled.value import CompiledValueFilter, EmptyCompiledName
    from .common import state_cache_kw, AggregateCompiledName, infer_descriptor

    @state_cache_kw
    def _get_cached_name(self: CompiledValueFilter, name, is_empty=False, is_descriptor=False):
        if is_empty:
            obj = self.compiled_value.access_handle.access._obj
            attr, is_get_descriptor = getattr_static(obj, name)
            if is_get_descriptor:
                if value := infer_descriptor(attr, self.compiled_value.parent_context):
                    return value.name
            return EmptyCompiledName(self._inference_state, name)
        return AggregateCompiledName((self.compiled_value, name))
    CompiledValueFilter._get_cached_name = _get_cached_name


# Patch BaseName._get_docstring to try remove the signature from the docstring
# as it were retrieved by py__doc__ so we don't get duplicate signatures.
def patch_BaseName_get_docstring():
    from jedi.api.classes import Completion, BaseName
    import re

    @inline
    def match_signature(doc: str):
        return re.compile(r"^.*?\(.*?\)(?: -> .*?)?[\n]").match

    @inline
    def match_rst_signature(doc: str):
        return re.compile(r"^\.\.\s?\w+::\s?.*[\n]?").match

    def _get_docstring(self: Completion):
        doc = BaseName._get_docstring(self)
        if match := match_signature(doc) or match_rst_signature(doc):
            doc = doc[match.end():].strip()
        return doc

    Completion._get_docstring = _get_docstring


# The pip version of Jedi uses custom typeshed stubs. There are several
# issues with them so we're using canon typeshed instead. We will run into
# compatibility issues down the road. This attemts to address those.
def patch_canon_typeshed_compatibility():
    from jedi.inference.gradual.typeshed import _IMPORT_MAP

    _IMPORT_MAP.clear()


# Patches ClassMixin.get_signatures to also check ``__new__`` and not just
# ``__init__`` for parameter names in signatures. Fixes partial(|<-  ) and
# probably others.
def patch_Script_get_signatures():
    from jedi.inference.value.klass import ClassMixin
    from .common import NoArguments

    def get_signatures(self: ClassMixin):
        if metaclasses := self.get_metaclasses():
            if sigs := self.get_metaclass_signatures(metaclasses):
                return sigs

        for instance in self.py__call__(NoArguments):
            for method_name in ("__call__", "__init__"):
                for method in instance.py__getattribute__(method_name):
                    ret = []
                    found = False
                    for sig in method.get_signatures():
                        sig = sig.bind(self)
                        found |= bool(sig.get_param_names())
                        ret += sig,
                    if found:
                        return ret
        return []
    ClassMixin.get_signatures = get_signatures


# Patches HelperValueMixin.is_same_class to work with classes that are for all
# intents and purposes identical, but are two distinct objects in memory.
# This happens when builtin stub values are cached.
def patch_HelperValueMixin_is_same_class():
    from jedi.inference.base_value import HelperValueMixin

    def is_same_class(self: HelperValueMixin, class2):
        # Class matching should prefer comparisons that are not this function.
        if type(class2).is_same_class != HelperValueMixin.is_same_class:
            return class2.is_same_class(self)

        # Compare by tree nodes.
        if self.tree_node is class2.tree_node is not None:
            return True
        return self == class2
    
    HelperValueMixin.is_same_class = is_same_class


# Patches Completion._complete_trailer to fix "object. |<-" completion.
# The added space causes Jedi to pick the wrong ``previous_leaf``, the dot
# operator, which ends up throwing an AssertionError during inference.
# This is a shorthand fix. The correct fix would be to detect the wrong leaf
# in Completion._complete_python, but that's too much code to drag in here.1 
def patch_Completion_complete_trailer():
    from jedi.api.completion import Completion, infer_call_of_leaf

    def _complete_trailer(self: Completion, previous_leaf):
        while previous_leaf.value == ".":
            previous_leaf = previous_leaf.get_previous_leaf()
        inferred_context = self._module_context.create_context(previous_leaf)
        values = infer_call_of_leaf(inferred_context, previous_leaf)

        # NOTE: ``cached_name`` so-called optimization is removed.
        # We've already optimized Jedi to hell.
        return None, self._complete_trailer_for_values(values)

    Completion._complete_trailer = _complete_trailer


# Patches BaseName._get_signatures to return the signatures of the first value
# of a set of values. The rest of the values are almost always duplicates.
def patch_BaseName_get_signatures():
    from jedi.inference.gradual.conversion import _python_to_stub_names
    from jedi.api.classes import BaseName, MixedName

    @inline
    def is_mixed_name(name):
        return MixedName.__instancecheck__
    
    def _get_signatures(self: BaseName, for_docstring=False):
        api_type = self._name.api_type

        if api_type == 'property':
            return []

        elif api_type == "statement" and for_docstring and not self.is_stub():
            return []

        elif is_mixed_name(self._name):
            return self._name.infer_compiled_value().get_signatures()

        for name in _python_to_stub_names((self._name,), fallback_to_python=True):
            for value in name.infer():
                if signatures := value.get_signatures():
                    return signatures
        return []

    BaseName._get_signatures = _get_signatures
