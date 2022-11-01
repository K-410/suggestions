from ... import types, utils
from bpy import types as bpy_types
import sys

_context = utils._context
bpy_struct = bpy_types.bpy_struct

ANSI_YELLOW     = '\033[33m'
ANSI_RED        = '\033[31m'
ANSI_GREEN      = '\033[32m'
ANSI_DEFAULT    = '\033[0m'


def red(*args: str) -> str:
    items = [str(s) for s in args]
    return f"{ANSI_RED}{' '.join(items)}{ANSI_DEFAULT}"

def yellow(*args: str) -> str:
    items = [str(s) for s in args]
    return f"{ANSI_YELLOW}{' '.join(items)}{ANSI_DEFAULT}"

def green(*args: str) -> str:
    items = [str(s) for s in args]
    return f"{ANSI_GREEN}{' '.join(items)}{ANSI_DEFAULT}"


# Global context members are fetched via bpy.types.Context.bl_rna.properties.
# Non-global members uses dynamic lookup (ctx_data_pointer_get in context.c),
# which means they have to be added manually.

# Obviously this sucks, but the complete listing can be found here:
# https://docs.blender.org/api/current/bpy.context.html

# Avoid getattr(context, member) as the main resolution fallback, because it
# may be NoneType, which is less useful than getting no completion at all.

# Except for Screen Context, dynamic members will never show up in context,
# because doing so would add nearly 100 entries which may not even exist.
buttons_context = {
    "texture_slot":             bpy_types.TextureSlot,
    "scene":                    bpy_types.Scene,
    "world":                    bpy_types.World,
    "object":                   bpy_types.Object,
    "mesh":                     bpy_types.Mesh,
    "armature":                 bpy_types.Armature,
    "lattice":                  bpy_types.Lattice,
    "curve":                    bpy_types.Curve,
    "meta_ball":                bpy_types.MetaBall,
    "light":                    bpy_types.Light,
    "speaker":                  bpy_types.Speaker,
    "lightprobe":               bpy_types.LightProbe,
    "camera":                   bpy_types.Camera,
    "material":                 bpy_types.Material,
    "material_slot":            bpy_types.MaterialSlot,
    "texture":                  bpy_types.Texture,
    "texture_user":             bpy_types.ID,
    "texture_user_property":    bpy_types.Property,
    "bone":                     bpy_types.Bone,
    "edit_bone":                bpy_types.EditBone,
    "pose_bone":                bpy_types.PoseBone,
    "particle_system":          bpy_types.ParticleSystem,
    "particle_system_editable": bpy_types.ParticleSystem,
    "particle_settings":        bpy_types.ParticleSettings,
    "cloth":                    bpy_types.ClothModifier,
    "soft_body":                bpy_types.SoftBodyModifier,
    "fluid":                    None,  # Not implemented
    "collision":                bpy_types.CollisionModifier,
    "brush":                    bpy_types.Brush,
    "dynamic_paint":            bpy_types.DynamicPaintModifier,
    "line_style":               bpy_types.FreestyleLineStyle,
    "collection":               bpy_types.LayerCollection,
    "gpencil":                  bpy_types.GreasePencil,
    "curves":                   None,  # Not implemented
    "volume":                   bpy_types.Volume,
}

clip_context = {
    "edit_movieclip":   bpy_types.MovieClip,
    "edit_mask":        bpy_types.Mask
}

file_context = {
    "active_file":               bpy_types.FileSelectEntry,
    "selected_files":       list[bpy_types.FileSelectEntry],
    "asset_library_ref":         bpy_types.AssetLibraryReference,
    "selected_asset_files": list[bpy_types.FileSelectEntry],
    "id":                        bpy_types.ID,
}

image_context = {
    "edit_image":       bpy_types.Image,
    "edit_mask":        bpy_types.Mask,
}

node_context = {
    "selected_nodes": list[bpy_types.Node],
    "active_node":         bpy_types.Node,
    "light":               bpy_types.Light,
    "material":            bpy_types.Material,
    "world":               bpy_types.World,
}

screen_context = {
    "scene":                            bpy_types.Scene,
    "visible_objects":             list[bpy_types.Object],
    "selectable_objects":          list[bpy_types.Object],
    "selected_objects":            list[bpy_types.Object],
    "selected_editable_objects":   list[bpy_types.Object],
    "editable_objects":            list[bpy_types.Object],
    "objects_in_mode":             list[bpy_types.Object],
    "objects_in_mode_unique_data": list[bpy_types.Object],
    "visible_bones":               list[bpy_types.EditBone],
    "editable_bones":              list[bpy_types.EditBone],
    "selected_bones":              list[bpy_types.EditBone],
    "selected_editable_bones":     list[bpy_types.EditBone],
    "visible_pose_bones":          list[bpy_types.PoseBone],
    "selected_pose_bones":         list[bpy_types.PoseBone],
    "selected_pose_bones_from_active_object": list[bpy_types.PoseBone],
    "active_bone":                      bpy_types.EditBone,
    "active_pose_bone":                 bpy_types.PoseBone,
    "active_object":                    bpy_types.Object,
    "object":                           bpy_types.Object,
    "edit_object":                      bpy_types.Object,
    "sculpt_object":                    bpy_types.Object,
    "vertex_paint_object":              bpy_types.Object,
    "weight_paint_object":              bpy_types.Object,
    "image_paint_object":               bpy_types.Object,
    "particle_edit_object":             bpy_types.Object,
    "pose_object":                      bpy_types.Object,
    "active_sequence_strip":            bpy_types.Sequence,
    "sequences":                   list[bpy_types.Sequence],
    "selected_sequences":          list[bpy_types.Sequence],
    "selected_editable_sequences": list[bpy_types.Sequence],
    "active_nla_track":                 bpy_types.NlaTrack,
    "active_nla_strip":                 bpy_types.NlaStrip,
    "selected_nla_strips":         list[bpy_types.NlaStrip],
    "selected_movieclip_tracks":   list[bpy_types.MovieTrackingTrack],
    "gpencil_data":                     bpy_types.GreasePencil,
    "gpencil_data_owner":               bpy_types.ID,
    "annotation_data":                  bpy_types.GreasePencil,
    "annotation_data_owner":            bpy_types.ID,
    "active_gpencil_layer":        list[bpy_types.GPencilLayer],
    "active_annotation_layer":          bpy_types.GPencilLayer,
    "active_gpencil_frame":        list[bpy_types.GPencilLayer],
    "visible_gpencil_layers":      list[bpy_types.GPencilLayer],
    "editable_gpencil_layers":     list[bpy_types.GPencilLayer],
    "editable_gpencil_strokes":    list[bpy_types.GPencilStroke],
    "active_operator":                  bpy_types.Operator,
    "active_action":                    bpy_types.Action,
    "selected_visible_actions":    list[bpy_types.Action],
    "selected_editable_actions":   list[bpy_types.Action],
    "editable_fcurves":            list[bpy_types.FCurve],
    "visible_fcurves":             list[bpy_types.FCurve],
    "selected_editable_fcurves":   list[bpy_types.FCurve],
    "selected_visible_fcurves":    list[bpy_types.FCurve],
    "active_editable_fcurve":           bpy_types.FCurve,
    "selected_editable_keyframes": list[bpy_types.Keyframe],
    "asset_library_ref":                bpy_types.AssetLibraryReference,
    "ui_list":                          bpy_types.UIList,
}

sequencer_context = {
    "edit_mask": bpy_types.Mask,
}

text_context = {
    "edit_text": bpy_types.Text
}

view3d_context = {
    "active_object": bpy_types.Object,
    "selected_ids": list[bpy_types.ID],
}


dyn_context = (
    buttons_context   |
    clip_context      |
    file_context      |
    image_context     |
    node_context      |
    screen_context    |
    sequencer_context |
    text_context      |
    view3d_context
)


# bpy.types.Property instances that get coerced to python types.
rna_pytypes = {
    "STRING": "",     # StringProperty
    "ENUM":   "",     # EnumProperty
    "BOOLEAN": False, # BoolProperty
    "INT":     0,     # IntProperty
    "FLOAT": 0.0,     # FloatProperty
}


def patch_inference_state_process(restore=False):
    from jedi.inference.compiled.subprocess import _InferenceStateProcess as cls

    if restore and (org_fn := getattr(cls, "_backup", False)):
        cls.get_or_create_access_handle = org_fn
        del cls._backup

    elif not restore:
        is_bpy_struct = bpy_types.bpy_struct.__instancecheck__

        def wrapper(self, obj, *, func=cls.get_or_create_access_handle):
            if is_bpy_struct(obj) and not isinstance(obj, RnaResolver):
                return func(self, RnaResolver(obj))
            else: return func(self, obj)

        cls._backup = cls.get_or_create_access_handle
        cls.get_or_create_access_handle = wrapper


from types import BuiltinMethodType

propcoll_keys = [k for k in dir(bpy_types.bpy_prop_collection)]
bpy_struct_keys = [k for k in dir(bpy_struct) if not k.startswith("__")]
default_code = types.noop.__code__.replace
resolver_py__dict__ = {}
stub_cache = {}
resolver_cache = {}

struct_id = bpy_types.ID.bl_rna.functions['copy'].parameters['id'].fixed_type

def stub_from_rna_func(parent_rna, func):
    # print(yellow("stub_from_rna_func"), parent_rna, func)
    try:
        return stub_cache[parent_rna, func]
    except:
        pass
    inputs = []
    restype = None
    # try:
    #     print("gen stub", func.identifier)
    # except BaseException:
    #     print_traceback()  # TODO: Remove if OK

    for p in func.parameters:
        # TODO: support additional semantics like:
        # - is_required
        # - is_never_none        (?)
        # - is_argument_optional (?)
        # Example:
        # argument1: (required)
        # argument2: (optional)
        if p.is_output:
            # All ID types use identical 'copy' function with bpy.types.ID as
            # the return type. This isn't really useful so we hard map it to
            # the resolver this function was found at
            if func.identifier == "copy" and p.fixed_type == struct_id:
                mapping = resolver_py__dict__[RnaResolver(parent_rna)]
            else:
                mapping = resolver_py__dict__[RnaResolver(p.fixed_type)]
            restype = type("restype", (), mapping)
        else:
            inputs.append(p.identifier)

    def stub() -> restype: pass
    stub.__name__ = func.identifier
    stub.__code__ = default_code(co_argcount=len(inputs), co_varnames=tuple(inputs))
    stub.__doc__ = func.description
    stub_cache[parent_rna, func] = stub
    return stub


class PropCollResolver:
    """Resolves bpy_prop_collection subclasses"""
    __class__ = list

    def __init__(self, rna):
        # def propcoll_iter() -> rna.fixed_type:
        #     pass

        class cls(list):
            __qualname__ = (rna.srna or rna.rna_type).identifier
            __module__ = "module"

        self.rna = rna
        self.cls = cls
        self.__iter__ = stub_from("__iter__", (), rna.fixed_type)
        
        for func in getattr(rna.srna, "functions", ()):
            setattr(cls, func.identifier, stub_from_rna_func(rna, func))

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except:
            # print(yellow("getting PCS", attr))
            try:
                ret = getattr(self.cls, attr)
                return ret
            except:
                print(red("failed getting PCS", attr))


    def __iter__(self):
        return iter((self.rna.fixed_type,))

    def __dir__(self):
        rna = self.rna
        *ret, = type(rna.bl_rna).__dict__.keys()
        if srna := rna.srna:
            ret += list(srna.functions.keys())
            ret += list(srna.properties.keys())
        return ret + propcoll_keys


def resolve_property_rna(rna):
    rna_type = rna.type

    if rna_type == 'POINTER':  # Property is a pointer to an RNA type
        return RnaResolver(rna.fixed_type)

    elif rna_type == 'COLLECTION':  # Property is a pointer to a property collection
        return PropCollResolver(rna)
    else:
        return rna_pytypes[rna_type]


def resolve_rna(resolver, rna, attr):
    if isinstance(rna, bpy_types.Property):
        ret = resolve_property_rna(rna)
        print(green("prop"), attr, "   ", ret)

    elif isinstance(rna, bpy_types.Function):
        stub = stub_from_rna_func(resolver, rna)
        ret = RnaResolver(rna)
        ret.__annotations__ = stub.__annotations__
        ret.__call__ = stub.__call__
        print(green("func"), attr, "   ", ret)

    elif isinstance(rna, (types.Callable, BuiltinMethodType)):
        ret = rna
        print(green("meth"), attr, "   ", ret)
    else:
        print(yellow(f"unhandled: {attr}"))
        return rna

    return ret


class bpy_typesResolver:
    """bpy.types resolver returns the RnaResolver version of any rna so that
    rna types can be used as annotations and still give completions as if the
    object was an instance."""
    def __getattribute__(self, attr):
        if attr.startswith("__"):
            return super().__getattribute__
        # print("get::", attr)
        obj = getattr(bpy_types, attr)
        # print("obj::", obj)
        ret = RnaResolver(obj.bl_rna)
        # print("got::", ret)
        return ret

    __dir__ = bpy_types.__dir__


resolver_rna = {}
bpy_typesResolver = bpy_typesResolver()

# Dictionary that holds bpy.types.Struct.bl_rna members
struct_rna_dict = {}

for attr in dir(bpy_types.Struct.bl_rna):
    struct_rna_dict[attr] = getattr(bpy_types.Struct.bl_rna, attr)

class StructRnaForwarder:
    """Forwards uncached bl_rna. This is needed because RnaResolver is cached
    and uses the same hash key as the rna it performs lookup on, which will
    always return itself.
    """

    def __new__(cls, rna):
        assert type(rna) is not RnaResolver
        self = super().__new__(cls)
        self.bl_rna = rna.bl_rna
        self._dict = bpy_struct.__dict__ | type(rna).__dict__ | struct_rna_dict
        # self.__dict__.update(bpy_struct.__dict__ | type(rna).__dict__ | struct_rna_dict)
        return self

    def __dir__(self):
        return dir(self.bl_rna) + list(self._dict.keys())
        # return dir(self._rna) + list(self.__dict__.keys())


class RnaResolver(bpy_struct):
    """Resolves bpy_struct subclasses' properties and methods into a format
    jedi understands without resorting to direct access. This is safer and
    much faster.
    RnaResolver expects 
    """
    def __new__(cls, rna, *, cache=resolver_cache):
        # Needed so pointer types resolve to their non-pointer types.
        if isinstance(rna, bpy_types.PointerProperty):
            rna = rna.fixed_type

        try:
            return cache[rna]
        except:  # Assume KeyError
            pass

        assert type(rna) is not RnaResolver


        resolver = cache[rna] = super().__new__(cls, rna)
        print("new resolver:", object.__str__(rna))
        print(object.__str__(rna))
        resolver_rna[resolver] = rna

        resolver.__mro__ = type(rna).__mro__

        # Instance rna vs struct rna is generally a headache to tell apart,
        # because both are instances of bpy_struct, but only one of them
        # contains introspection attributes.
        if not hasattr(rna, "properties"):
            rna = rna.bl_rna

        # Add rna subclass methods/properties
        res_dict = dict(rna.properties.items() + rna.functions.items())
        print(*res_dict.keys(), sep=" ")
        # Add struct rna and non-rna methods/properties
        res_dict.update(bpy_struct.__dict__ | type(rna).__dict__)

        resolver_py__dict__[resolver] = res_dict
        return resolver

    def __getattribute__(self, attr: str):
        assert self in resolver_py__dict__, type.__repr__(self)
        assert isinstance(self, RnaResolver)
        print(yellow("get", attr), object.__str__(self))

        if attr == "bl_rna":
            return StructRnaForwarder(resolver_rna[self].bl_rna)

        if rna := resolver_py__dict__[self].get(attr):
            return resolve_rna(self, rna, attr)
        # print("not rna:", attr)
        try:
            return super().__getattribute__(attr)
        except:
            pass

        # Nasty, but this allows jedi to annotate the rna type
        restype = type("restype", (), resolver_py__dict__[self])
        if attr == "__annotations__":
            return {"return": restype}
        if attr == "__call__":
            def test() -> restype:
                pass
            return test
        # print(red(f"get failed: {repr(attr)}", self))
        raise AttributeError

    def __dir__(self):
        print(green("dir"), resolver_py__dict__[self].keys())
        return resolver_py__dict__[self].keys()


def stub_from(name, arglist, restype, doc=""):
    def stub() -> restype:
        pass

    stub.__name__ = name
    stub.__code__ = default_code(co_argcount=len(arglist), co_varnames=arglist)
    stub.__doc__ = doc
    return stub


class ContextResolver(RnaResolver):
    """ContextResolver gives a more complete list of members by attempting
    to read the real context before using RnaResolver as the fallback.
    """

    def __new__(cls):
        self = super().__new__(cls, bpy_types.Context.bl_rna)
        resolver_py__dict__[self]["copy"] = stub_from("copy", (), dict)
        return self
        # try:
        #     return resolver_cache[bpy_types.Context.bl_rna]
        # except:
        #     super().____(self, bpy_types.Context)
        #     # Add dict as the return type for bpy_types.Context.copy()
        #     resolver_py__dict__[self]["copy"] = stub_from("copy", (), dict)

    def __getattr__(self, attr: str, *, c=_context):
        # Dynamic context attributes are not listed in completions because it
        # clutters the list with over 100 entries that may or may not exist.
        # However they still can be resolved when typing manually.
        try:
            ret = dyn_context[attr]
            try:
                if isinstance(ret.bl_rna, bpy_struct):
                    ret = RnaResolver(ret.bl_rna)
                    # print("resolving", ret)
            except:
                pass
            return ret
        except:
            return getattr(c, attr)

    def __dir__(self, *, c=_context):
        return list(set(list(screen_context.keys()) + list(RnaResolver.__dir__(self))))
        # return list(set(list(dir(c)) + list(RnaResolver.__dir__(self))))



# Return type mapping for bpy_struct methods - manually edited.
# Not ideal, but bpy_struct does not have a blueprint, and its
# methods are used in all StructRNA instances.
def set_bpy_struct_mapping():
    import idprop
    import typing

    # IDPropertyUIManager isn't exposed, so we do this hack by adding a
    # StringProperty on a bpy.types.ID instance.
    uid = hex(id(utils.this_module()))
    _context.window_manager[uid] = ""
    IDPropertyUIManager = type(_context.window_manager.id_properties_ui(uid))
    _context.window_manager.pop(uid)

    bpy_struct_rtype_mapping = {
        "as_pointer":               int,
        "bl_rna_get_subclass":      bpy_types.Struct,
        "bl_rna_get_subclass_py":   bpy_struct,
        "driver_add":               bpy_types.FCurve | list[bpy_types.FCurve],
        "driver_remove":            bool,
        "get":                      typing.Any,
        "id_properties_clear":      None,
        "id_properties_ensure":     idprop.types.IDPropertyGroup,
        "id_properties_ui":         IDPropertyUIManager, 
        "is_property_hidden":       bool,
        "is_property_overridable_library": bool,
        "is_property_readonly":     bool,
        "is_property_set":          bool,
        "items":                    idprop.types.IDPropertyGroupViewItems,
        "keyframe_delete":          bool,
        "keyframe_insert":          bool,
        "keys":                     idprop.types.IDPropertyGroupViewKeys,
        "path_from_id":             str,
        "path_resolve":             bpy_types.bpy_prop,
        "pop":                      typing.Any,
        "property_overridable_library_set": bool,
        "property_unset":           None,
        "type_recast":              bpy_struct,
        "values":                   idprop.types.IDPropertyGroupViewValues
    }

    for identifier, rtype in bpy_struct_rtype_mapping.items():
        try:
            doc = getattr(bpy_struct, identifier).__doc__
        except AttributeError:
            print("Warning: textension.suggestions bpy_struct API mismatch")
            continue

        def func() -> rtype:
            pass
        func.__doc__ = doc
        setattr(RnaResolver, identifier, func)


def patch_get_builtin_module_names():
    """Patches jedi's builtin modules getter to include file-less modules
    like _bpy, that doesn't exist in sys.builtin_module_names"""
    from jedi.inference.compiled.subprocess import functions

    def get_builtin_module_names(inference_state):
        extended = []
        for modname, module in sys.modules.items():
            if not hasattr(module, "__file__"):
                extended.append(modname)
        return tuple(set(sys.builtin_module_names) | set(extended))

    functions.get_builtin_module_names.__code__ = get_builtin_module_names.__code__


def patch_anonymous_param():
    """Patch dynamic params so bpy.types.Operator method parameters are
    automatically inferred.
    """
    from textension.plugins.suggestions.resolvers import RnaResolver, ContextResolver
    from jedi.inference.dynamic_params import dynamic_param_lookup
    from jedi.inference.compiled.access import create_access
    from jedi.inference.compiled.value import CompiledValue
    from jedi.inference.value.instance import BoundMethod
    from jedi.inference.base_value import ValueSet, NO_VALUES
    from bpy import types as bpy_types

    def bpy_dynamic_param(function_value, param_index,
        RnaResolver=RnaResolver,
        ContextResolver=ContextResolver,
        create_access=create_access,
        CompiledValue=CompiledValue,
        BoundMethod=BoundMethod,
        ValueSet=ValueSet,
        NO_VALUES=NO_VALUES,
        Operator=bpy_types.Operator,
        Event=bpy_types.Event):

        if isinstance(function_value, BoundMethod):
            func_name = function_value.name.string_name
            if func_name in {"invoke", "execute", "modal", "draw", "poll"}:

                # Support mixin so that bpy.types.Operator can appear anywhere
                for base in function_value.instance.class_value.py__bases__():
                    for val in base.infer():
                        try:
                            obj = val.access_handle.access._obj
                        except:
                            continue
                        if isinstance(obj.bl_rna.bl_rna, Operator):
                            if param_index == 1:
                                obj = ContextResolver()
                                print("resolving context")
                            else:
                                obj = RnaResolver(Event.bl_rna)
                            access = create_access(function_value.inference_state, obj)
                            c = CompiledValue(function_value.inference_state, access, None)
                            return ValueSet({c})
                            # This must exist so python binds it as a free variable, which
                            # matches the number of free variables in the original function.
                            # Must not be used as the variable may be *anything*
                            bpy_types  # Intentional
        return NO_VALUES

    dynamic_param_lookup.__code__ = bpy_dynamic_param.__code__
    dynamic_param_lookup.__defaults__ = bpy_dynamic_param.__defaults__


def patch_getattr_paths():
    """Patch getattr_paths so any rna type is intercepted and resolved using
    RnaResolver.
    """
    from jedi.inference.compiled.access import DirectObjectAccess, _sentinel
    from warnings import catch_warnings, simplefilter
    from inspect import ismodule, getmodule
    import builtins

    def getattr_paths(self, name, default=_sentinel):
        # This part is identical to the start of the original function
        try:
            with catch_warnings(record=True):
                simplefilter("always")
                return_obj = getattr(self._obj, name)
        except Exception as e:
            if default is _sentinel:
                if isinstance(e, AttributeError):
                    raise
                raise AttributeError
            return_obj = default

        # This part redirects rna resolutions
        if isinstance(return_obj, bpy_struct):
            if isinstance(return_obj, bpy_types.Context):
                return_obj = ContextResolver()
            else:
                return_obj = RnaResolver(return_obj)
        elif return_obj is bpy_types:
            return_obj = bpy_typesResolver

        # This part is identical to the rest of the original function
        access = self._create_access(return_obj)
        if ismodule(return_obj):
            return [access]

        try:
            module = return_obj.__module__
        except AttributeError:
            pass
        else:
            if module is not None and isinstance(module, str):
                try:
                    __import__(module)
                    # For some modules like _sqlite3, the __module__ for classes is
                    # different, in this case it's sqlite3. So we have to try to
                    # load that "original" module, because it's not loaded yet. If
                    # we don't do that, we don't really have a "parent" module and
                    # we would fall back to builtins.
                except ImportError:
                    pass

        module = getmodule(return_obj)
        if module is None:
            module = getmodule(type(return_obj))
            if module is None:
                module = builtins
        return [self._create_access(module), access]

    DirectObjectAccess.getattr_paths = getattr_paths