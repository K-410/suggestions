"""This module implements blf module inference."""

from .. import tools
import blf


def apply():
    tools._add_rtype_overrides(rtype_data)


rtype_data = (
    (blf.dimensions, tuple[float, float]),
    (blf.load, int),
)