# This module implements RNA type inference.

from jedi.inference.compiled.value import CompiledValueFilter, SignatureParamName
from jedi.inference.value.instance import ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.signature import AbstractSignature
from jedi.inference.compiled import builtin_from_name

from itertools import chain, repeat
from inspect import Parameter

from textension.utils import _context, _forwarder, inline

from ._mathutils import float_vector_map
from ..common import VirtualFilter, VirtualInstance, VirtualName, VirtualValue, get_mro_dict, state_cache
from ..tools import runtime, state, make_compiled_value, make_instance

import bpy
import _bpy
import types

from bpy_types import RNAMeta, StructRNA


rnadef_types = bpy.types.Property, bpy.types.Function
rna_types = RNAMeta, StructRNA


def apply():
    patch_AnonymousParamName_infer()


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


callable_types = (
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.BuiltinFunctionType,
    types.FunctionType,
    types.MethodDescriptorType
)

context_rna_pointer = _context.as_pointer()
_void = object()


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


rna_py_type_map = {
    "STRING": str,
    "ENUM": str,
    "INT": int,
    "BOOLEAN": bool,
    "FLOAT": float
}


def get_context_instance():
    if not runtime.context_rna:
        runtime.context_rna = ContextInstance(bpy.types.Context.bl_rna)
    return runtime.context_rna


def get_rna_value(obj, parent_value):
    if isinstance(obj, bpy.props._PropertyDeferred):
        return PropertyValue((getattr(bpy.types, obj.function.__name__), parent_value))
    if not isinstance(obj, type) and obj.as_pointer() == context_rna_pointer:
        return get_context_instance()
    return RnaValue((obj.bl_rna, parent_value))


get_dict = type.__dict__["__dict__"].__get__
get_mro = type.__dict__["__mro__"].__get__


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


# TODO: Is this needed?
class PropertyFilter(VirtualFilter):
    pass


# TODO: Is this needed?
class PropertyInstance(VirtualInstance):
    pass


# TODO: Is this needed?
class PropertyValue(VirtualValue):
    filter_cls   = PropertyFilter
    instance_cls = PropertyInstance


def rnadef_to_value(rnadef, parent):
    if is_bpy_func(rnadef):
        return RnaFunction((rnadef, parent))
    
    assert is_bpy_struct(rnadef), f"rnadef_to_value failed for {rnadef} ({parent})"

    type = rnadef.type

    if type == 'POINTER':
        return RnaValue((rnadef.fixed_type, parent))
    elif type == 'COLLECTION':
        return PropCollectionValue((rnadef, parent))
    
    assert type in rna_py_type_map, f"rnadef_to_value failed for {rnadef} ({parent}), type: {type}"

    # ``INT``, ``FLOAT`` and ``BOOLEAN`` can be vectors.
    if type not in {'STRING', 'ENUM'} and rnadef.array_length != 0:
        if rnadef.subtype in float_vector_map:
            return make_compiled_value(float_vector_map[rnadef.subtype], parent.as_context())
            # return MathutilsValue((float_vector_map[rnadef.subtype], parent))
        return PropArrayValue((rna_py_type_map[type], parent))
    return builtin_from_name(state, rna_py_type_map[type].__name__)


rna_fallbacks = {}


# TODO: Implement using callbacks.
def add_Object_member_fallbacks():
    items = rna_fallbacks["Object"] = {}
    items["children"] = list[bpy.types.Object]
    items["children_recursive"] = list[bpy.types.Object]


add_Object_member_fallbacks()



def get_id_data(value: "RnaValue"):
    while value and isinstance(value, RnaValue) and not is_id(value.obj):
        value = value.parent_value
    return value


def rna_fallback_value(parent, name):
    if overrides := rna_fallbacks.get(parent.name.string_name):
        if rtype := overrides.get(name):
            # TODO: Still needs implementation.
            from jedi.inference.value.iterable import FakeTuple
            from jedi.inference.lazy_value import LazyKnownValue
            v = LazyKnownValue(RnaValue((bpy.types.Object, parent)))
            return ValueSet([FakeTuple(state, [v])])


# An RnaName doesn't have to infer to an RnaValue. It just means the name
# is an RnaValue member name.
class RnaName(VirtualName):
    def infer(self):
        parent = self.parent_value
        name  = self.string_name
        obj = parent.members[name]

        if isinstance(parent, ContextInstance) and isinstance(obj, type):
            value = RnaValue((obj.bl_rna, parent))

        elif isinstance(obj, rnadef_types):
            value = rnadef_to_value(obj, parent) or make_compiled_value(obj, parent.as_context())

        elif name == "id_data":
            value = get_id_data(parent)
            # If this fails, some rna values aren't struct definitions.
            assert value, f"Unhandled id_data object: {obj} ({name})"

        elif tmp := rna_fallback_value(parent, name):
            value = tmp

        else:
            if name == "bl_rna":
                value = RnaValue((obj, parent))
            else:
                value = make_compiled_value(obj, parent.as_context())
                print(f"RnaName: Unhandled member '{parent.name.string_name}.{name}'")
            return ValueSet((value,))
        return value.py__call__(None)

    def __repr__(self):
        return f"{repr(self.parent_value)}.{self.string_name}"


# This class implements the RNA value lookup.
class RnaFilter(VirtualFilter):
    name_cls = RnaName

    @state_cache
    def values(self):
        parent = self.compiled_value
        bl_rna = parent.obj

        if instanced := self.is_instance or bl_rna != bl_rna.bl_rna:
            # Instances get everything
            members = parent.members

        else:
            # Non-instances get bpy.types.Struct members and class members.
            members  = RnaValue((bpy.types.Struct.bl_rna, parent)).members
            members |= get_mro_dict(bl_rna)

        data = zip(repeat(parent), members, repeat(instanced))
        return list(map(RnaName, data))

    def __repr__(self):
        return f"RnaFilter({self.compiled_value.obj.identifier})"


# Represents an instance of an RnaValue or RnaFunction.
class RnaInstance(VirtualInstance):
    obj = _forwarder("class_value.obj")

    @property
    def api_type(self):
        if isinstance(self.class_value.obj, bpy.types.Function):
            return "function"
        return super().api_type

    # For eg ``bpy.data.objects[0].copy().``.
    # XXX: Arguments aren't used for now.
    def py__call__(self, arguments):
        if self.api_type == "function":

            # bpy_struct.copy() has ``bpy.types.ID`` as its return type,
            # so we need to map it to the real value.
            if self.obj.identifier == "copy":
                return self.class_value.parent_value.py__call__(None)

            for rnadef in self.class_value.obj.parameters:
                if rnadef.is_output:
                    return rnadef_to_value(rnadef, self.class_value).py__call__(None)
        print("RnaInstance.py__call__ returned nothing for", self)
        return NO_VALUES

    # Implements subscript for bpy_prop_collections.
    def py__simple_getitem__(self, index):
        if isinstance(self.class_value, PropCollectionValue):
            instance = RnaValue((self.class_value.obj.fixed_type, self.class_value)).instance
            return ValueSet((instance,))
        return NO_VALUES

    def get_signatures(self):
        if isinstance(self.class_value, RnaFunction):
            return self.class_value.get_signatures()
        return ()

    def __repr__(self):
        return f"RnaInstance({self.class_value.obj.identifier})"


# RNA value created from struct RNA definition.
class RnaValue(VirtualValue):
    filter_cls   = RnaFilter
    instance_cls = RnaInstance

    @property
    def api_type(self):
        if isinstance(self.obj, bpy.types.Function):
            return "function"
        return "unknown"

    def get_signatures(self):
        if self.api_type == "function":
            return [RnaFunctionSignature(self, is_bound=False)]
        return ()

    def py__doc__(self):
        if not (doc := self.obj.description):
            print(f"Missing ``description`` for {self.rna}")
        return doc

    def py__name__(self):
        return self.obj.identifier

    @property
    @state_cache
    def members(self):
        rna = self.obj

        # It's possible we're accessing a real instance like bpy.data.
        # ``bl_rna`` is self-referencing so this is the safest way.
        if rna != rna.bl_rna:
            print("passing RnaValue as real rnas!:", rna, "vs", rna.bl_rna)
        rna = rna.bl_rna

        return get_mro_dict(rna) | dict(chain(rna.functions.items(), rna.properties.items()))

    def __repr__(self):
        return f"RnaValue({self.obj.identifier})"

    def py__mro__(self):
        assert is_bpy_struct(self.obj)
        context = self.as_context()

        yield self

        for obj in type(self.obj.bl_rna).__mro__:
            if is_bpy_struct_subclass(obj) and obj is not StructRNA:
                yield RnaValue((obj.bl_rna, self))
            else:
                yield make_compiled_value(obj, context)

    def py__doc__(self):
        return self.obj.description


# TODO: Is this needed?
class RnaFunctionFilter(VirtualFilter):
    pass


class RnaFunction(RnaValue):
    filter_cls = RnaFunctionFilter

    def get_signatures(self):
        return (RnaFunctionSignature(self, self, is_bound=True),)

    def get_param_names(self):
        ret = []
        rna = self.obj
        if rna.use_self:
            p = RnaFunctionParamName(value=self, name="self")
            ret += [p]

        for param in self.obj.parameters:
            if not param.is_output:
                ret += [RnaFunctionParamName(value=self, param_rna=param,)]
        return ret

    @property
    @state_cache
    def members(self):
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
        if self._param is not None:
            return self._param.identifier
        return self._string_name

    @property
    def annotated(self):
        return getattr(self._param, "fixed_type", _void)

    @property
    def default(self):
        if self._param.is_required:
            return ""
        return f'={getattr(self._param, "default", "")}'

    def get_kind(self):
        return getattr(Parameter, self.kind_name)

    def infer(self):
        if (annotated := self.annotated) is not _void:
            return ValueSet((make_compiled_value(annotated, self.get_root_context()),))
        return NO_VALUES

    def to_string(self):
        s = self.string_name
        if self.annotated is not _void:
            import inspect
            fmt = inspect.formatannotation(type(self.annotated))
            s += ": " + fmt.replace("bpy_types", "bpy.types")
        return s + self.default

    def get_root_context(self):
        return self._value.parent_context.get_root_context()


class RnaFunctionSignature(AbstractSignature):
    def __init__(self, value, function_value=None, is_bound=False):
        self.value = value
        self.is_bound = is_bound
        self._function_value = function_value or value

    def bind(self, value):
        return RnaFunctionSignature(value, self._function_value, is_bound=True)


class MathutilsValue(VirtualValue):
    def get_filters(self, is_instance=False, origin_scope=None):
        yield CompiledValueFilter(state, self, is_instance)


# bpy.types.Object.[bound_box | rotation_axis_angle | etc.]
class PropArrayValue(VirtualValue):
    # Implements subscript for bpy_prop_array types.
    def py__simple_getitem__(self, index):
        return ValueSet((make_instance(self.obj, self.parent_context),))

    # Implements for loop variable inference.
    def py__iter__(self, contextualized_node=None):
        return ValueSet((LazyKnownValues(self.py__simple_getitem__(None)),))

    @property
    @state_cache
    def members(self):
        return get_mro_dict(bpy.types.bpy_prop_array)


class PropCollectionFilter(VirtualFilter):
    name_cls = RnaName


class PropCollectionValue(RnaValue):
    filter_cls = PropCollectionFilter

    @property
    @state_cache
    def members(self):
        mapping = get_mro_dict(bpy.types.bpy_prop_collection)
        if srna := self.obj.srna:
            mapping |= dict(chain(srna.functions.items(), srna.properties.items()))
        return mapping


class ContextInstance(RnaInstance):
    parent_context = _forwarder("_parent_context")

    # The context is static so make members an attribute.
    members: dict = None

    def __init__(self, obj):
        self._arguments  = None
        self.filter = ContextInstanceFilter((self, True))
        from ._bpy import Importer_redirects

        module = Importer_redirects["bpy"]
        self._parent_context = module.as_context()

        class_value = RnaValue((obj, module))
        context_members = class_value.members.copy()

        for member in _bpy.context_members()["screen"]:
            attr, is_sequence = context_type_map.get(member, ("", False))
            if obj := getattr(bpy.types, attr, None):
                context_members[member] = obj if not is_sequence else list[obj]
            else:
                pass  # Could print a warning. This is a bug.

        self.members = context_members
        self.class_value = class_value

    def get_filters(self, *_, **__):
        yield self.filter

    def _get_value_filters(self, *_):
        return (self.filter,)


class ContextInstanceFilter(RnaFilter):
    def values(self):
        ret = super().values()
        data = zip(repeat(self.compiled_value), _bpy.context_members()["screen"])
        return ret + list(map(RnaName, data))


# Patch jedi's anonymous parameter inference so that bpy.types.Operator
# method parameters can be automatically inferred.
def patch_AnonymousParamName_infer():
    from jedi.inference.names import AnonymousParamName, NO_VALUES, ValueSet
    from jedi.inference.value.instance import BoundMethod

    is_RnaValue = RnaValue.__instancecheck__
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
                return RnaValue((cls, parent_value))
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
            if not is_RnaValue(value):
                continue

            parent = value
            if param := infer_rna_param(parent.obj, func_name, param_index):
                if param.identifier == "context":
                    instance = get_context_instance()
                else:
                    instance, = rnadef_to_value(param, parent).py__call__(None)
                return ValueSet((instance,))

        return NO_VALUES
    AnonymousParamName.infer = infer
