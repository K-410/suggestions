"""This module implements RNA type inference."""

from jedi.inference.compiled.value import CompiledValue, SignatureParamName
from jedi.inference.value.iterable import FakeList, FakeTuple, LazyKnownValue
from jedi.inference.lazy_value import LazyKnownValues, NO_VALUES
from jedi.inference.signature import AbstractSignature

from itertools import repeat
from operator import attrgetter
from inspect import Parameter

from textension.utils import _context, _forwarder, inline, starchain, get_dict, get_mro_dict

from ._mathutils import float_subtypes, MathutilsValue
from ..common import Values, NoArguments, get_type_name
from ..tools import runtime, state, make_compiled_value, make_instance

import bpy
import _bpy
import types
from .. import common

from bpy_types import RNAMeta, StructRNA


rnadef_types = bpy.types.Property, bpy.types.Function
rna_types = RNAMeta, StructRNA

bpy_struct_magic = {}
rna_fallbacks = {}

context_rna_pointer = _context.as_pointer()
_void = object()


def apply():
    patch_AnonymousParamName_infer()

    from ._bpy import Importer_redirects
    runtime.context_rna = ContextInstance(bpy.types.Context.bl_rna)
    runtime.data_rna = RnaInstance(get_rna_value(_bpy.data.bl_rna, Importer_redirects["bpy"]))

    # Map bpy_struct magic methods.
    for cls in reversed(StructRNA.__mro__):
        mapping = get_dict(cls)
        for key, value in mapping.items():
            doc = getattr(value, "__doc__", None)
            if not doc and key in bpy_struct_magic:
                continue
            bpy_struct_magic[key] = value

    add_Object_member_fallbacks()

    from mathutils import Vector, Matrix
    common.virtual_overrides[Vector] = MathutilsValue
    common.virtual_overrides[Matrix] = MathutilsValue


@inline
def is_bpy_func(obj) -> bool:
    return bpy.types.Function.__instancecheck__


@inline
def is_bpy_struct(obj) -> bool:
    return bpy.types.bpy_struct.__instancecheck__


@inline
def is_id(obj) -> bool:
    return bpy.types.ID.__instancecheck__


@inline
def is_bpy_struct_subclass(obj) -> bool:
    return bpy.types.bpy_struct.__subclasscheck__


@inline
def is_property_deferred(obj) -> bool:
    return bpy.props._PropertyDeferred.__instancecheck__


@inline
def is_operator_subclass(obj) -> bool:
    return bpy.types.Operator.__subclasscheck__


@inline
def is_string(obj) -> bool:
    return str.__instancecheck__


@inline
def is_virtual_value(obj) -> bool:
    return common.VirtualValue.__instancecheck__


@inline
def is_blend_data(obj) -> bool:
    return bpy.types.BlendData.__instancecheck__


@inline
def is_compiled_value(obj) -> bool:
    return CompiledValue.__instancecheck__


@inline
def get_bpy_type(obj):
    return bpy.types.__getattribute__


@inline
def get_rna_dict(rna) -> dict:
    get_rnadefs = attrgetter("functions", "properties")
    prop_collection_items = bpy.types.bpy_prop_collection.items
    def get_rna_dict(rna) -> dict:
        return dict(starchain(map(prop_collection_items, get_rnadefs(rna))))
    return get_rna_dict


# See doc/python_api/sphinx_doc_gen.py.
context_type_map = {
    # context_member: (RNA type, is_collection)
    "active_action": ("Action", False),
    "active_annotation_layer": ("GPencilLayer", False),
    "active_bone": ("EditBone", False),
    "active_file": ("FileSelectEntry", False),
    "active_gpencil_frame": ("GPencilFrame", True),  # Was GreasePencilLayer (wrong).
    "active_gpencil_layer": ("GPencilLayer", True),
    "active_node": ("Node", False),
    "active_object": ("Object", False),
    "active_operator": ("Operator", False),
    "active_pose_bone": ("PoseBone", False),
    "active_sequence_strip": ("Sequence", False),
    "active_editable_fcurve": ("FCurve", False),
    "active_nla_strip": ("NlaStrip", False),
    "active_nla_track": ("NlaTrack", False),
    "annotation_data": ("GreasePencil", False),
    "annotation_data_owner": ("ID", False),
    "armature": ("Armature", False),
    "asset_library_ref": ("AssetLibraryReference", False),
    "bone": ("Bone", False),
    "brush": ("Brush", False),
    "camera": ("Camera", False),
    "cloth": ("ClothModifier", False),
    "collection": ("LayerCollection", False),
    "collision": ("CollisionModifier", False),
    "curve": ("Curve", False),
    "dynamic_paint": ("DynamicPaintModifier", False),
    "edit_bone": ("EditBone", False),
    "edit_image": ("Image", False),
    "edit_mask": ("Mask", False),
    "edit_movieclip": ("MovieClip", False),
    "edit_object": ("Object", False),
    "edit_text": ("Text", False),
    "editable_bones": ("EditBone", True),
    "editable_gpencil_layers": ("GPencilLayer", True),
    "editable_gpencil_strokes": ("GPencilStroke", True),
    "editable_objects": ("Object", True),
    "editable_fcurves": ("FCurve", True),
    "fluid": ("FluidSimulationModifier", False),
    "gpencil": ("GreasePencil", False),
    "gpencil_data": ("GreasePencil", False),
    "gpencil_data_owner": ("ID", False),
    "curves": ("Hair Curves", False),
    "id": ("ID", False),
    "image_paint_object": ("Object", False),
    "lattice": ("Lattice", False),
    "light": ("Light", False),
    "lightprobe": ("LightProbe", False),
    "line_style": ("FreestyleLineStyle", False),
    "material": ("Material", False),
    "material_slot": ("MaterialSlot", False),
    "mesh": ("Mesh", False),
    "meta_ball": ("MetaBall", False),
    "object": ("Object", False),
    "objects_in_mode": ("Object", True),
    "objects_in_mode_unique_data": ("Object", True),
    "particle_edit_object": ("Object", False),
    "particle_settings": ("ParticleSettings", False),
    "particle_system": ("ParticleSystem", False),
    "particle_system_editable": ("ParticleSystem", False),
    "pointcloud": ("PointCloud", False),
    "pose_bone": ("PoseBone", False),
    "pose_object": ("Object", False),
    "scene": ("Scene", False),
    "sculpt_object": ("Object", False),
    "selectable_objects": ("Object", True),
    "selected_asset_files": ("FileSelectEntry", True),
    "selected_bones": ("EditBone", True),
    "selected_editable_actions": ("Action", True),
    "selected_editable_bones": ("EditBone", True),
    "selected_editable_fcurves": ("FCurve", True),
    "selected_editable_keyframes": ("Keyframe", True),
    "selected_editable_objects": ("Object", True),
    "selected_editable_sequences": ("Sequence", True),
    "selected_files": ("FileSelectEntry", True),
    "selected_ids": ("ID", True),
    "selected_nla_strips": ("NlaStrip", True),
    "selected_movieclip_tracks": ("MovieTrackingTrack", True),
    "selected_nodes": ("Node", True),
    "selected_objects": ("Object", True),
    "selected_pose_bones": ("PoseBone", True),
    "selected_pose_bones_from_active_object": ("PoseBone", True),
    "selected_sequences": ("Sequence", True),
    "selected_visible_actions": ("Action", True),
    "selected_visible_fcurves": ("FCurve", True),
    "sequences": ("Sequence", True),
    "soft_body": ("SoftBodyModifier", False),
    "speaker": ("Speaker", False),
    "texture": ("Texture", False),
    "texture_slot": ("TextureSlot", False),
    "texture_user": ("ID", False),
    "texture_user_property": ("Property", False),
    "ui_list": ("UIList", False),
    "vertex_paint_object": ("Object", False),
    "view_layer": ("ViewLayer", False),
    "visible_bones": ("EditBone", True),
    "visible_gpencil_layers": ("GPencilLayer", True),
    "visible_objects": ("Object", True),
    "visible_pose_bones": ("PoseBone", True),
    "visible_fcurves": ("FCurve", True),
    "weight_paint_object": ("Object", False),
    "volume": ("Volume", False),
    "world": ("World", False),
}

global_context = set(
    rnadef.identifier for rnadef in bpy.types.Context.bl_rna.properties)

extended_context = context_type_map.keys() | global_context

rna_py_types = {
    "STRING":  str,
    "ENUM":    str,
    "INT":     int,
    "BOOLEAN": bool,
    "FLOAT":   float
}


property_instance_map = {
    "StringProperty": str,
    "FloatProperty":  float,
    "IntProperty":    int,
    "BoolProperty":   bool,
    "EnumProperty":   str,
    "PointerProperty": None,
    "CollectionProperty": None,
    "FloatVectorProperty": None,  # Needs bpy_prop_array
    "IntVectorProperty": None,
    "BoolVectorProperty": None,
}


def get_rna_value(obj, parent_value):
    if is_property_deferred(obj):
        return common.VirtualValue((get_bpy_type(obj.function.__name__), parent_value))

    # ``bpy.context`` interception.
    if not isinstance(obj, type) and obj.as_pointer() == context_rna_pointer:
        return runtime.context_rna

    # ``bpy.data`` interception.
    if is_blend_data(obj) and obj != obj.bl_rna:
        return runtime.data_rna

    return RnaValue((obj.bl_rna, parent_value))


def rnadef_to_value(rnadef, parent):
    if is_bpy_func(rnadef):
        return RnaFunction((rnadef, parent))
    
    type = rnadef.type

    if type == 'POINTER':
        return get_rna_value(rnadef.fixed_type, parent)

    elif type == 'COLLECTION':
        return PropCollectionValue((rnadef, parent))
    
    # Possible vector type strings: INT, FLOAT, BOOLEAN
    if type not in {'STRING', 'ENUM'} and (rnadef.is_array or rnadef.array_length):

        if rnadef.subtype in float_subtypes:
            return MathutilsValue((float_subtypes[rnadef.subtype], parent))

        return PropArrayValue((rna_py_types[type], parent))

    return common.get_builtin_value(rna_py_types[type].__name__)


# TODO: Implement using callbacks.
def add_Object_member_fallbacks():
    items = rna_fallbacks["Object"] = {}
    items["children"] = list[bpy.types.Object]
    items["children_recursive"] = list[bpy.types.Object]


def get_rna_id_data(value: "RnaValue"):
    while value and is_rna_value(value) and not is_id(value.obj):
        value = value.parent_value
    return value


def rna_fallback_value(parent, name):
    if overrides := rna_fallbacks.get(parent.name.string_name):
        if rtype := overrides.get(name):
            # TODO: Still needs implementation.
            v = LazyKnownValue(get_rna_value(bpy.types.Object, parent))
            return Values((FakeTuple(state, [v]),))


# An RnaName doesn't have to infer to an RnaValue. It just means the name
# is an RnaValue member name.
class RnaName(common.VirtualName):
    def py__doc__(self):
        name = self.string_name

        # ``bl_rna`` as a rnadef is defined on bpy.types.Struct.
        if name == "bl_rna":
            return bpy.types.Struct.bl_rna.description

        # Context members have no rna definition, so we need to get them manually.
        elif name in extended_context:

            if self.parent_value is runtime.context_rna.class_value:
                # Global context members.
                if name in global_context:
                    value = rnadef_to_value(bpy.types.Context.bl_rna.properties[name], self.parent_value)
                    if is_rna_value(value):
                        return value.obj.description
                    return value.py__doc__()

                # All other context members.
                attr, is_sequence = context_type_map[name]
                if obj := getattr(bpy.types, attr, None):
                    rna = getattr(obj, "bl_rna", None)
                    if is_sequence:
                        obj = list[obj]
                    if not rna:
                        return make_compiled_value(attr, self.parent_context).py__doc__()
                    return rna.description

        for value in self.parent_value.py__mro__():
            if is_rna_value(value):
                obj = value.obj
                if is_prop_collection_value(value) and obj.srna:
                    obj = obj.srna

                # The member is defined in rna.
                if rnadef := get_rna_dict(obj).get(name):
                    return rnadef.description
                
                # The member is defined on the type or any of its bases.
                if member := get_mro_dict(obj).get(name):
                    if doc := getattr(member, "__doc__", None):
                        if is_string(doc):
                            return doc

            # The member is defined in bpy_struct.
            elif is_compiled_value(value) and name in bpy_struct_magic:
                if value.access_handle.access._obj is StructRNA:
                    member = bpy_struct_magic[name]

                    if is_string(member):
                        return member
                    if doc := getattr(member, "__doc__", None):
                        if is_string(doc):
                            return doc
        return ""


# Represents an instance of an RnaValue or RnaFunction.
class RnaInstance(common.VirtualInstance):
    obj = _forwarder("class_value.obj")

    @property
    def api_type(self):
        if is_bpy_func(self.class_value.obj):
            return "function"
        return super().api_type

    # For eg ``bpy.data.objects[0].copy().``.
    # XXX: Arguments aren't used for now.
    def py__call__(self, arguments):
        if self.api_type == "function":

            # bpy_struct.copy() has ``bpy.types.ID`` as its return type,
            # so we need to map it to the real value.
            if self.obj.identifier == "copy":
                return self.class_value.parent_value.py__call__(arguments)

            for rnadef in self.class_value.obj.parameters:
                if rnadef.is_output:
                    return rnadef_to_value(rnadef, self.class_value).py__call__(arguments)
        print("RnaInstance.py__call__ returned nothing for", self)
        return NO_VALUES

    # Implements subscript for bpy_prop_collections.
    def py__simple_getitem__(self, index):
        if is_prop_collection_value(self.class_value):
            srna = self.class_value.obj.fixed_type
            if srna == bpy.types.Property.bl_rna:
                value = RnaPropertyComposite((srna, self.class_value))
            else:
                value = get_rna_value(srna, self.class_value)
            return Values((value.as_instance(),))
        return NO_VALUES

    def py__getitem__(self, index_value_set, contextualized_node):
        return self.py__simple_getitem__(0)

    def get_signatures(self):
        if isinstance(self.class_value, RnaFunction):
            return self.class_value.get_signatures()
        return ()

    def __repr__(self):
        return f"RnaInstance({self.class_value.obj.identifier})"


# RNA value created from struct RNA definition.
class RnaValue(common.VirtualValue):
    instance_cls = RnaInstance

    def infer_name(self, name: common.VirtualName):
        name_str = name.string_name
        members = self.members

        if name_str not in members:
            # print(f"RnaValue.infer_name(): '{name}' not in members of {self}.")
            return NO_VALUES

        obj = members[name_str]

        if is_rna_context_instance(self) and isinstance(obj, type):
            if isinstance(obj, types.GenericAlias):
                v, = get_rna_value(obj.__args__[0].bl_rna, self).py__call__(NoArguments)
                return Values((FakeList(state, [LazyKnownValue(v)]),))

            value = get_rna_value(obj.bl_rna, self)

        elif isinstance(obj, rnadef_types) and obj.rna_type != bpy.types.Struct.bl_rna:
            value = rnadef_to_value(obj, self)

        elif name_str == "id_data":
            value = get_rna_id_data(self)

        elif tmp := rna_fallback_value(self, name_str):
            value = tmp

        elif name_str == "bl_rna":
            value = get_rna_value(bpy.types.Struct, self)

        else:
            if callable(obj):
                return Values((make_compiled_value(obj, self.as_context()),))
            
            # Potentially magic methods. This is not worth inferring.
            return NO_VALUES
        return value.py__call__(NoArguments)

    @property
    def api_type(self):
        if self.obj.rna_type.name == "Function Definition":
            return "function"
        
        # Is the bl_rna member.
        elif is_virtual_value(self.parent_value) and self.obj == self.parent_value.obj:
            return "instance"

        elif self.is_class():
            return "class"
        elif self.is_instance():
            return "instance"
        return "unknown"

    def get_signatures(self):
        if self.api_type == "function":
            return [RnaFunctionSignature(self, is_bound=False)]
        return ()

    def py__doc__(self):
        if not (doc := self.obj.description):
            print(f"Missing ``description`` for {self.obj}")
        return doc

    def py__name__(self):
        return self.obj.identifier

    def get_members(self):
        return get_mro_dict(self.obj) | get_rna_dict(self.obj)

    def __repr__(self):
        return f"{get_type_name(self.__class__)}({self.obj.identifier})"

    @common.state_cache
    def py__mro__(self):
        context = self.as_context()
        ret = [self]

        for obj in self.obj.bl_rna.__class__.__mro__[1:]:
            if is_bpy_struct_subclass(obj) and obj is not StructRNA:
                ret += get_rna_value(obj.bl_rna, self),
            else:
                ret += make_compiled_value(obj, context),
        return ret

    def py__doc__(self):
        obj = self.obj
        if is_operator_subclass(obj.__class__) and is_string(obj.__doc__):
            return obj.__doc__
        if obj.name:
            return f"{obj.name}\n\n{obj.description}"
        return obj.description

    @common.state_cache
    def get_filter_values(self, is_instance):
        bl_rna = self.obj

        if instanced := is_instance or bl_rna != bl_rna.bl_rna:
            # Instances get everything
            members = self.members

        else:
            # Non-instances get bpy.types.Struct members and class members.
            members  = get_rna_value(bpy.types.Struct.bl_rna, self).members
            members |= get_mro_dict(bl_rna)

        return list(map(RnaName, zip(repeat(self), members, repeat(instanced))))


@inline
def is_rna_value(obj) -> bool:
    return RnaValue.__instancecheck__


# Some fixed type RNA definitions use the base. Property is one of those.
# This mashes Property subclasses together to provide useful completions,
# otherwise we won't be able complete stuff like ``enum_items``.
class RnaPropertyComposite(RnaValue):
    def get_members(self):
        comp = {}
        for cls in self.obj.bl_rna.__class__.__subclasses__():
            comp |= get_rna_dict(cls.bl_rna)
        return comp


class RnaFunction(RnaValue):
    def get_signatures(self):
        return (RnaFunctionSignature(self, self, is_bound=True),)

    def get_param_names(self):
        ret = []
        rna = self.obj
        if rna.use_self:
            p = RnaFunctionParamName(value=self, name="self")
            ret += p,

        for param in self.obj.parameters:
            if not param.is_output:
                ret += RnaFunctionParamName(value=self, param_rna=param,),
        return ret

    def get_members(self):
        return get_mro_dict(bpy.types.bpy_func)


class RnaFunctionParamName(SignatureParamName):
    def __init__(self, value, param_rna=None, name=""):
        self._param = param_rna
        self._value = value
        self._string_name = name
        self.kind_name = "POSITIONAL_ONLY"

        if param_rna and not param_rna.is_required:
            self.kind_name = "KEYWORD_ONLY"

    @property
    def string_name(self):
        if paramdef := self._param:
            return paramdef.identifier
        return self._string_name

    @property
    def annotated(self):
        paramdef = self._param
        param_value = rnadef_to_value(paramdef, self._value)

        if is_virtual_value(param_value):
            restype = param_value.obj

            if is_bpy_struct(restype):
                restype = type(restype)
            return restype

        elif param_value.get_root_context().is_builtins_module():
            if ret := __builtins__.get(param_value.name.string_name):
                return ret
        return getattr(paramdef, "fixed_type", _void)

    @property
    def default(self):
        param = self._param
        if param.is_required or param.is_never_none:
            return ""

        # Optional enum flags are empty sets.
        if param.is_enum_flag:
            s = "set()"
        else:
            s = getattr(param, "default", "None")
        return f' = {s}'

    def get_kind(self):
        return getattr(Parameter, self.kind_name)

    def infer(self):
        if (annotated := self.annotated) is not _void:
            return Values((make_compiled_value(annotated, self.get_root_context()),))
        print(f"RnaFunctionParamName.infer(): {self.string_name} could not be inferred on {self._value}.")
        return NO_VALUES

    def to_string(self):
        s = self.string_name
        default = self.default
        
        if annotated := self.annotated:
            if is_bpy_struct(annotated):
                annotated = type(annotated)
            import inspect
            fmt = inspect.formatannotation(annotated)
            s += ": " + fmt.replace("bpy_types", "bpy.types")
        return s + default

    @inline
    def get_root_context(self):
        return _forwarder("_value.parent_context.get_root_context")


class RnaFunctionSignature(AbstractSignature):
    def __init__(self, value, function_value=None, is_bound=False):
        self.value = value
        self.is_bound = is_bound
        self._function_value = function_value or value

    def bind(self, value):
        return RnaFunctionSignature(value, self._function_value, is_bound=True)

    @property
    def annotation_string(self):
        rnadef = self.value.obj
        outputs = []

        # Multiple outputs means the function returns a sequence e.g a tuple.
        for param in rnadef.parameters:
            if param.is_output:
                outputs += param,

        if outputs:
            tmp = []
            for param in outputs:
                param_value = rnadef_to_value(param, self.value)
                if is_virtual_value(param_value):
                    restype = param_value.obj
                    if is_bpy_struct(restype):
                        restype = type(restype)
                    import inspect
                    ann = inspect.formatannotation(restype)
                    if "bpy_types" in ann:
                        ann = ann.replace("bpy_types", "bpy.types")
                else:
                    if param_value.get_root_context().is_builtins_module():
                        ann = param_value.name.string_name
                    else:
                        print("failed", param, rnadef)  # Debug.
                        continue
                tmp += ann,

            elems = ", ".join(tmp)
            if len(tmp) > 1:
                return f"Sequence[{elems}]"
            return elems
        return "None"


# bpy.types.Object.[bound_box | rotation_axis_angle | etc.]
class PropArrayValue(common.VirtualValue):
    # Implements subscript for bpy_prop_array types.
    def py__simple_getitem__(self, index):
        return Values((make_instance(self.obj, self.parent_context),))

    # Implements for loop variable inference.
    def py__iter__(self, contextualized_node=None):
        return Values((LazyKnownValues(self.py__simple_getitem__(None)),))

    def get_members(self):
        return get_mro_dict(bpy.types.bpy_prop_array)


class PropCollectionValue(RnaValue):
    def get_members(self):
        mapping = get_mro_dict(bpy.types.bpy_prop_collection)
        if self.obj.srna:
            mapping |= get_rna_dict(self.obj.srna)
        return mapping


# For user-defined property collections, i.e CollectionProperty.
class IdPropCollectionValue(common.VirtualValue):
    values = NO_VALUES
    
    def infer_name(self, name: common.VirtualName):
        name_str = name.string_name

        # Currently only add/get are inferred from assigned values.
        if name_str in {"add", "get"}:
            return self.values.infer()

        obj = self.members[name_str]
        return Values((make_compiled_value(obj, self.as_context()),))

    def py__call__(self, arguments):
        self.values = arguments
        return Values((self.instance_cls(self, arguments),))

    def py__simple_getitem__(self, index):
        for value in self.values.infer():
            return value.py__call__(NoArguments)
        return NO_VALUES


class ContextInstance(RnaInstance):
    parent_context = _forwarder("_parent_context")

    # The context is static so make members an attribute.
    members: dict = None

    def __init__(self, obj):
        self._arguments  = None
        from ._bpy import Importer_redirects

        module = Importer_redirects["bpy"]
        self._parent_context = module.as_context()

        class_value = get_rna_value(obj, module)
        context_members = class_value.members.copy()

        for member in _bpy.context_members()["screen"]:
            attr, is_sequence = context_type_map.get(member, ("", False))
            if obj := getattr(bpy.types, attr, None):
                context_members[member] = obj if not is_sequence else list[obj]
            else:
                pass  # Could print a warning. This is a bug.

        self.members = context_members
        self.class_value = class_value

    def _get_value_filters(self, *_):
        yield from self.get_filters()

    def get_filters(self, **kw):
        yield common.VirtualFilter((self, True))

    def get_filter_values(self, is_instance):
        data = zip(repeat(self.class_value), _bpy.context_members()["screen"])
        return self.class_value.get_filter_values(True) + list(map(RnaName, data))

    def get_filter_get(self, name_str, _):
        if ret := self.class_value.get_filter_get(name_str, True):
            return ret
        
        # These aren't part of the screen context.
        if name_str in context_type_map:
            return Values((NonScreenContextName((self.class_value, name_str, True)),))
        return ()


@inline
def is_rna_context_instance(obj) -> bool:
    return ContextInstance.__instancecheck__


@inline
def is_prop_collection_value(obj) -> bool:
    return PropCollectionValue.__instancecheck__


class NonScreenContextName(RnaName):
    def infer(self):
        name, is_collection = context_type_map[self.string_name]
        cls = getattr(bpy.types, name)

        value, = get_rna_value(cls.bl_rna, self.parent_value).py__call__(NoArguments)
        if is_collection:
            value = FakeList(state, (LazyKnownValue(value),))
        return Values((value,))


# Patch jedi's anonymous parameter inference so that bpy.types.Operator
# method parameters can be automatically inferred.
def patch_AnonymousParamName_infer():
    from jedi.inference.value.instance import BoundMethod
    from jedi.inference.names import AnonymousParamName, NO_VALUES
    from ..common import Values

    infer_orig = AnonymousParamName.infer

    def infer_rna_param(rna_obj, func_name, param_index):
        if rna_func := rna_obj.bl_rna.functions.get(func_name):
            index = 0
            for param in rna_func.parameters:
                if not param.is_output:
                    if index == param_index:
                        return param
                    index += 1
        return None

    def rna_value_from_tree_value(value, parent_value):
        if value.is_compiled():
            return None

        for c in (c for b in value.py__bases__() for c in b.infer()):
            if not c.is_compiled():
                continue
            if not c.access_handle.access._obj is bpy.types.bpy_struct:
                continue
            cls_name = value.tree_node.children[1].value
            if cls := getattr(bpy.types, cls_name, None):
                return get_rna_value(cls, parent_value)
        return None

    def infer(self: AnonymousParamName):
        if ret := infer_orig(self):
            return ret

        func = self.function_value

        if not isinstance(func, BoundMethod):
            return NO_VALUES

        func_name = func._wrapped_value.tree_node.children[1].value
        param = self.tree_name.parent

        # An RNA function doesn't include self in its parameters list.
        param_index = param.position_index - int(bool(func.is_bound_method()))
        context = self.parent_context

        for value in (v for b in func.instance.class_value.py__bases__()
                        for v in b.infer()):

            # Support inferring Operator from ``bpy_types.py``. Since this is
            # a tree node, we have to convert it into an RnaValue.
            if rna := rna_value_from_tree_value(value, context):
                value = rna

            # Only RnaValue parameters are inferred for now.
            if not is_rna_value(value):
                continue

            parent = value
            if param := infer_rna_param(parent.obj, func_name, param_index):
                if param.identifier == "context":
                    instance = runtime.context_rna
                else:
                    instance, = rnadef_to_value(param, parent).py__call__(NoArguments)
                return Values((instance,))

        return NO_VALUES
    AnonymousParamName.infer = infer
