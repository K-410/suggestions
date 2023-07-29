# This module implements import resolution for bpy.

from jedi.inference.names import SubModuleName
from jedi.inference.syntax_tree import tree_name_to_values
import bpy
import _bpy

from types import ModuleType
from itertools import repeat
from operator import getitem

from ._mathutils import float_subtypes
from ._bpy_types import MathutilsValue, PropArrayValue, IdPropCollectionValue, NO_VALUES, get_rna_value
from ..common import VirtualFunction, VirtualValue, VirtualModule, Importer_redirects, CompiledModule_redirects, find_definition, AggregateValues, state


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


def _as_module(obj, name):
    m = ModuleType(name)
    try:
        m.__dict__.update(obj.__dict__)
    except AttributeError:
        for name in dir(obj):
            m.__dict__[name] = getattr(obj, name)
    return m


# Needs to exist for import redirects.
class BpyModule(VirtualModule):
    pass


class OpsModule(VirtualModule):
    def get_submodule_names(self, only_modules=False):
        sub = map(getitem, map(str.partition, _bpy.ops.dir(), repeat("_OT_")), repeat(0))
        names = set(map(str.lower, set(sub)))
        return list(map(SubModuleName, repeat(self.as_context()), names))


class PropsModule(VirtualModule):
    def py__getattribute__(self, name_or_str, **kw):
        if value := prop_func_map.get(name_or_str.value):
            return (value,)
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
        return IdPropCollectionValue((obj, parent)).py__call__(value)
    return NO_VALUES


class PropertyFunction(VirtualFunction):
    def py__call__(self, arguments):
        # TODO: This isn't ideal.
        instance = VirtualValue((bpy.props._PropertyDeferred, self)).as_instance(arguments)
        instance.py__call__ = lambda *_, **__: self.py_instance__call__(arguments)
        return AggregateValues((instance,))

    def py_instance__call__(self, arguments):
        func_name = self.obj.__name__

        if func_name == "CollectionProperty" in simple_property_types:
            return infer_collection_type(arguments)

        elif func_name in simple_property_types:
            obj = simple_property_types[func_name]
            return VirtualValue((obj, self)).py__call__(arguments)

        elif func_name in vector_property_types:
            return infer_vector_type(func_name, arguments, self)

        elif func_name == "PointerProperty":
            return infer_pointer_type(arguments)

        obj = getattr(bpy.types, func_name).bl_rna
        return get_rna_value(obj, self).py__call__(arguments)


def fix_bpy_imports():
    Importer_redirects["bpy"]     = BpyModule(bpy)
    Importer_redirects["bpy.ops"] = OpsModule(bpy.ops)

    # Add bpy.props redirect.
    Importer_redirects["bpy.props"] = PropsModule(bpy.props)
    Importer_redirects["_bpy.props"] = Importer_redirects["bpy.props"]

    # Add property function redirects.
    parent_value = Importer_redirects["bpy.props"]
    for name in prop_names:
        prop_func_map[name] = PropertyFunction((getattr(bpy.props, name), parent_value))

    # Add redirects for compiled value interceptions.
    for module in Importer_redirects.values():
        CompiledModule_redirects[module.obj] = module
