"""This module implements import resolution for bpy."""

from jedi.inference.names import SubModuleName
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.arguments import TreeArguments
import bpy
import _bpy

from types import ModuleType
from itertools import repeat
from operator import getitem

from ._mathutils import float_subtypes
from ._bpy_types import MathutilsValue, PropArrayValue, IdPropCollectionValue, NO_VALUES, get_rna_value
from ..common import VirtualFunction, VirtualInstance, VirtualValue, VirtualModule, Importer_redirects, CompiledModule_redirects, find_definition, Values, state, add_module_redirect, NoArguments
from ..tools import make_compiled_value
from textension.utils import inline_class


def apply():
    fix_bpy_imports()
    _add_collection_property()


prop_names = (
    "BoolProperty",
    "BoolVectorProperty",
    "CollectionProperty",
    "EnumProperty",
    "FloatProperty",
    "FloatVectorProperty",
    "IntProperty",
    "IntVectorProperty",
    "PointerProperty",
    "StringProperty",
)

# Map of PropertyFunctions for custom property inference.
prop_func_map: dict[str, "PropertyFunction"] = {}

simple_property_types = {
    "BoolProperty":   bool,
    "EnumProperty":   str,
    "FloatProperty":  float,
    "IntProperty":    int,
    "StringProperty": str,
}

vector_property_types = {
    "BoolVectorProperty",
    "FloatVectorProperty",
    "IntVectorProperty"
}


def _add_collection_property():
    # ``bpy_prop_collection_idprop`` is the base for CollectionProperty, but isn't
    # available in bpy.types. It is however a subclass of ``bpy_prop_collection``.
    for cls in bpy.types.bpy_prop_collection.__subclasses__():
        if cls.__name__ == "bpy_prop_collection_idprop":
            simple_property_types["CollectionProperty"] = cls
            return None
    else:
        # Shouldn't happen unless the api broke.
        print("Warning (Textension): _add_collection_property failed to find "
              "bpy_prop_collection_idprop. This is a bug.")


def _name_as_string(name_or_str):
    if isinstance(name_or_str, str):
        return name_or_str
    return name_or_str.value


@inline_class(bpy)
class bpy_module(VirtualModule):
    string_names = ("bpy",)
    def infer_name(self, name):
        # ``bpy.ops`` and ``bpy.props`` redirect to virtual modules.
        if name.string_name in {"ops", "props"}:
            value = Importer_redirects[f"bpy.{name.string_name}"]

        else:
            sentinel = object()
            obj = getattr(self.obj, name.string_name, sentinel)
            if obj is sentinel:
                return NO_VALUES

            context = None
            if not isinstance(obj, ModuleType):
                context = self.as_context()

            value = make_compiled_value(obj, context)
        return Values((value,))


class OperatorWrapper(VirtualFunction):
    def py__doc__(self):
        # We want an empty line between the signature and description.
        return self.obj.__doc__.replace("\n", "\n\n", 1)


# Wish we didn't need this, but bpy.ops uses a custom ``__getattribute__``
# to dynamically lookup submodules and operators. If jedi is in restricted
# descriptor access mode, completing bpy.ops is otherwise impossible.
class OpsSubModule(VirtualModule):
    # Placeholder and override base class' string_names property.
    # This is set by ``_create_ops_submodule``.
    string_names = ()

    def py__getattribute__(self, name_or_str, **kw):
        if ret := getattr(self.obj, _name_as_string(name_or_str)):
            if isinstance(ret, bpy.ops._BPyOpsSubModOp):
                return Values((make_compiled_value(ret, self.as_context()),))
        return NO_VALUES

    def py__doc__(self):
        submod_name = self.obj.__name__.split(".")[-1]
        return f"bpy.ops sub-module '{submod_name}'"

    def get_members(self):
        return dict.fromkeys(self.obj.__dir__())

    def infer_name(self, name):
        operator = OperatorWrapper((getattr(self.obj, name.string_name), self))
        return Values((operator,))


def _create_ops_submodule(module):
    submod = OpsSubModule(module)
    submod.string_names = tuple(module.__name__.split("."))  # ("bpy", "ops", "x")
    return submod


class OpsSubModuleName(SubModuleName):
    def py__doc__(self):
        return ""


@inline_class(bpy.ops)
class ops_module(VirtualModule):
    api_type = "module"
    string_names = ("bpy", "ops")

    # Only for import completions.
    def get_submodule_names(self, only_modules=False):
        sub = map(getitem, map(str.partition, _bpy.ops.dir(), repeat("_OT_")), repeat(0))
        names = set(map(str.lower, set(sub)))
        return list(map(OpsSubModuleName, repeat(self.as_context()), names))

    def py__getattribute__(self, name_or_str, **kw):
        name_or_str = _name_as_string(name_or_str)

        if ret := getattr(self.obj, name_or_str):
            if isinstance(ret, ModuleType) and ret.__name__ == "bpy.ops." + name_or_str:
                return Values((_create_ops_submodule(ret),))
        return super().py__getattribute__(name_or_str)

    def get_members(self):
        sub = map(getitem, map(str.partition, _bpy.ops.dir(), repeat("_OT_")), repeat(0))
        return dict.fromkeys(map(str.lower, set(sub)))

    def infer_name(self, name):
        name_str = name.string_name
        if ret := getattr(self.obj, name_str):
            if isinstance(ret, ModuleType) and ret.__name__ == "bpy.ops." + name_str:
                return Values((_create_ops_submodule(ret),))
        return super().infer_name(name)


@inline_class(bpy.props)
class bpy_props_module(VirtualModule):
    string_names = ("bpy", "props")
    def py__getattribute__(self, name_or_str, **kw):
        if value := prop_func_map.get(name_or_str.value):
            return Values((value,))
        return super().py__getattribute__(name_or_str)


def infer_vector_type(name, arguments, parent):
    if value := dict(arguments.unpack()).get("subtype"):
        if value.data.type == "string":
            if obj := float_subtypes.get(value.data._get_payload()):
                return MathutilsValue((obj, parent)).py__call__(None)

    obj = simple_property_types[name.replace("Vector", "")]
    return PropArrayValue((obj, parent)).py__call__(None)


def infer_pointer_type(arguments):
    if value := dict(arguments.unpack()).get("type"):
        # ``value.data`` could be a name or atom_expr.
        ref = value.data
        if ref.type == "atom_expr":
            return value.infer().py__call__(None)

        if namedef := find_definition(ref):
            for value in tree_name_to_values(state, value.context, namedef):
                return value.py__call__(None)
    return NO_VALUES


def infer_collection_type(arguments):
    if value := dict(arguments.unpack()).get("type"):
        obj = simple_property_types["CollectionProperty"]
        parent = value.context._value
        return IdPropCollectionValue((obj, parent)).virtual_call(value)
    return NO_VALUES


class PropertyFunction(VirtualFunction):
    def virtual_call(self, arguments=NoArguments, instance: VirtualInstance = None):

        if isinstance(instance, VirtualInstance) and isinstance(arguments, TreeArguments):
            property_name = instance.class_value.py__name__()

            if property_name == "CollectionProperty" in simple_property_types:
                return infer_collection_type(arguments)
            elif obj := simple_property_types.get(property_name):
                return VirtualValue((obj, self)).py__call__(arguments)
            elif property_name in vector_property_types:
                return infer_vector_type(property_name, arguments, self)
            elif property_name == "PointerProperty":
                return infer_pointer_type(arguments)

            else:
                obj = getattr(bpy.types, property_name).bl_rna
                return get_rna_value(obj, self).py__call__(arguments)
        return super().virtual_call(arguments, instance)


def fix_bpy_imports():
    add_module_redirect(bpy_module)
    add_module_redirect(ops_module)

    # Add bpy.props redirect.
    add_module_redirect(bpy_props_module)
    add_module_redirect(bpy_props_module, "_bpy.props")

    # Add property function redirects.
    for name in prop_names:
        prop_func_map[name] = PropertyFunction((getattr(bpy.props, name), bpy_props_module))

    # Add redirects for compiled value interceptions.
    for module in Importer_redirects.values():
        CompiledModule_redirects[module.obj] = module
