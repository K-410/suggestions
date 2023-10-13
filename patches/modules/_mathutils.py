"""This module implements mathutils type inference."""

from mathutils import Vector, Quaternion, Euler, Matrix, Color
from jedi.inference.gradual.base import GenericClass
from textension.utils import starchain
import itertools

from ..common import NO_VALUES, VirtualValue, VirtualInstance, Values
from .. import common
from .. import tools
import bpy


virtuals = (VirtualInstance, VirtualValue)

# Blender FloatProperty subtype-to-mathutils type mapping.
float_subtypes = {
    "ACCELERATION": Vector,
    "DIRECTION":    Vector,
    "TRANSLATION":  Vector,
    "VELOCITY":     Vector,
    "XYZ_LENGTH":   Vector,
    "XYZ":          Vector,
    "COORDINATES":  Vector,

    "MATRIX":       Matrix,
    "EULER":        Euler,
    "QUATERNION":   Quaternion,
    "COLOR":        Color,
    "COLOR_GAMMA":  Color
}


def infer_matmul(self: VirtualValue, arguments):
    this = self.obj
    for value in dict(arguments.unpack()).values():
        for value in value.infer():
            rtype = None
            if isinstance(value, VirtualInstance):
                value = value.class_value
            if not isinstance(value, VirtualValue):
                continue
            other = value.obj

            if this is Matrix and other in (Vector, Matrix):
                rtype = MathutilsValue((other, self))

            elif this is Vector and other is Vector:
                rtype = common.cached_builtins.float
                
            elif this is Quaternion and other in (Quaternion, Vector):
                rtype = MathutilsValue((other, self))

            if rtype:
                return rtype.py__call__(common.NoArguments)
    return NO_VALUES


class MathutilsValue(VirtualValue):

    # Needed so tuple(Vector()) is inferrable.
    def is_sub_class_of(self, class_value):
        if isinstance(class_value, GenericClass):
            if class_value._class_value.py__name__() == "Iterable":
                return True
        return False

    def py__simple_getitem__(self, index):
        if self.obj == Vector:
            return Values((common.cached_builtins.float,))
        elif self.obj == Matrix:
            return Values((MathutilsValue((Vector, self)),))
        return NO_VALUES

    def py__getattribute__(self, name, name_context=None, position=None, analysis_errors=True):
        if not isinstance(name, str):
            name = name.value

        obj = getattr(self.obj, name, ...)
        if obj is not ...:
            return Values((tools.make_compiled_value(obj, self.as_context()),))

        return super().py__getattribute__(name, name_context, position, analysis_errors)

    def virtual_call(self, arguments, instance=None):
        if instance and instance.class_value.obj.__name__ == "__matmul__":
            return infer_matmul(self, arguments)
        return super().virtual_call(arguments, instance)


# Support Matrix.row and Matrix.col subscript and iteration.
class MatrixAccessValue(VirtualValue):
    def py__simple_getitem__(self, index_unused):
        return Values((MathutilsValue((Vector, self)),))


rtype_data = (
    # mathutils.Vector
    (Vector.Fill, Vector),
    (Vector.Linspace, Vector),
    (Vector.Range, Vector),
    (Vector.Repeat, Vector),
    (Vector.copy, Vector),
    (Vector.cross, Vector | float),  # type:ignore
    (Vector.dot, float),
    (Vector.freeze, Vector),
    (Vector.lerp, Vector),
    (Vector.normalized, Vector),
    (Vector.orthogonal, Vector),
    (Vector.project, Vector),
    (Vector.reflect, Vector),
    (Vector.resized, Vector),
    (Vector.rotation_difference, Quaternion),
    (Vector.slerp, Vector),
    (Vector.to_2d, Vector),
    (Vector.to_3d, Vector),
    (Vector.to_4d, Vector),
    (Vector.to_track_quat, Quaternion),
    (Vector.to_tuple, tuple[float]),
    (Vector.__copy__, Vector),
    (Vector.__deepcopy__, float),
    (Vector.__getitem__, float),
    (Vector.__matmul__, float),

    # mathutils.Euler
    (Euler.__getitem__, float),
    (Euler.to_matrix, Matrix),
    (Euler.to_quaternion, Quaternion),
    (Euler.copy, Euler),
    (Euler.__copy__, Euler),
    (Euler.__deepcopy__, Euler),

    # mathutils.Matrix
    (Matrix.Diagonal, Matrix),
    (Matrix.Identity, Matrix),
    (Matrix.LocRotScale, Matrix),
    (Matrix.OrthoProjection, Matrix),
    (Matrix.Rotation, Matrix),
    (Matrix.Scale, Matrix),
    (Matrix.Shear, Matrix),
    (Matrix.Translation, Matrix),
    (Matrix.adjugated, Matrix),
    (Matrix.copy, Matrix),
    (Matrix.decompose, tuple[Vector, Quaternion, Vector]),
    (Matrix.freeze, Matrix),
    (Matrix.inverted, Matrix),
    (Matrix.inverted_safe, Matrix),
    (Matrix.lerp, Matrix),
    (Matrix.normalized, Matrix),
    (Matrix.resize_4x4, Matrix),
    (Matrix.to_2x2, Matrix),
    (Matrix.to_3x3, Matrix),
    (Matrix.to_4x4, Matrix),
    (Matrix.to_euler, Euler),
    (Matrix.to_quaternion, Quaternion),
    (Matrix.to_scale, Vector),
    (Matrix.to_translation, Vector),
    (Matrix.transposed, Matrix),

    # mathutils.Quaternion
    (Quaternion.conjugated, Quaternion),
    (Quaternion.copy, Quaternion),
    (Quaternion.cross, Quaternion),
    (Quaternion.dot, float),
    (Quaternion.freeze, Quaternion),
    (Quaternion.inverted, Quaternion),
    (Quaternion.normalized, Quaternion),
    (Quaternion.rotation_difference, Quaternion),
    (Quaternion.slerp, Quaternion),
    (Quaternion.to_axis_angle, tuple[Vector, float]),
    (Quaternion.to_euler, Euler),
    (Quaternion.to_exponential_map, Vector),
    (Quaternion.to_matrix, Matrix),
    (Quaternion.to_swing_twist, tuple[Quaternion, float]),

)

# Add Vector swizzle descriptor overrides.
vector_swizzles = starchain(
    zip(map(Vector.__dict__.__getitem__,
            map("".join, itertools.product("xyzw", repeat=i))),
        itertools.repeat(Vector))
    for i in (2, 3, 4)
)

descriptor_data = [
    # matutils.Vector
    (Vector.x, float),
    (Vector.y, float),
    (Vector.z, float),
    (Vector.w, float),
    *vector_swizzles,

    # matutils.Euler
    (Euler.x, float),
    (Euler.y, float),
    (Euler.z, float),
    (Euler.order, str),
    (Euler.is_wrapped, bool),
    (Euler.is_frozen, bool),
    (Euler.is_valid, bool),

    # matutils.Matrix
    (Matrix.is_frozen, bool),
    (Matrix.is_negative, bool),
    (Matrix.is_orthogonal, bool),
    (Matrix.is_orthogonal_axis_vectors, bool),
    (Matrix.is_valid, bool),
    (Matrix.is_wrapped, bool),

    # mathutils.Quaternion
    (Quaternion.angle, float),
    (Quaternion.axis, Vector),
    (Quaternion.is_frozen, bool),
    (Quaternion.is_valid, bool),
    (Quaternion.is_wrapped, bool),
    (Quaternion.magnitude, float),
    (Quaternion.w, float),
    (Quaternion.x, float),
    (Quaternion.y, float),
    (Quaternion.z, float),
]


# Versioning.
if _desc := getattr(Matrix, "is_identity", None):
    descriptor_data += (_desc, bool),


virtual_overrides = (
    *((x, MathutilsValue) for x in (Vector, Euler, Matrix, Quaternion)),
    *((x, MatrixAccessValue) for x in (Matrix.row, Matrix.col))
)


def apply():
    tools._add_rtype_overrides(rtype_data)
    tools._add_descriptor_overrides(descriptor_data)

    tools.add_virtual_overrides(virtual_overrides)
