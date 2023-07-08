# This module implements auto-completions.

from textension.utils import _context, _system
from textension import utils, ui

from operator import attrgetter, methodcaller
from textension.overrides import OpOverride
from textension.btypes.defs import OPERATOR_CANCELLED

import os
import sys
import bpy
import gc


# Backup.
_gc_enable = gc.enable

_get_sync_key = attrgetter("select_end_line", "select_end_character")

# Token separators excluding dot/period
separators = {*" !\"#$%&\'()*+,-/:;<=>?@[\\]^`{|}~"}

runtime = utils.namespace(loaded=False)
runtime = utils._get_dict(bpy.types.WindowManager).setdefault("_suggestions", runtime)


BLF_BOLD = 1 << 11  # ``blf.enable(0, BLF_BOLD)`` adds bold effect.


class TEXT_OT_autocomplete(OpOverride):
    def invoke(self):
        deferred_complete()
        return OPERATOR_CANCELLED


class Description(ui.widgets.TextView):
    def draw(self):
        return super().draw()


class Suggestions(ui.widgets.ListBox):
    st:  bpy.types.SpaceTextEditor
    _temp_lines             = []

    is_visible: bool        = False
    last_position           = (0, 0)
    sync_key                = ()

    background_color        = 0.15, 0.15, 0.15, 1.0
    border_color            = 0.30, 0.30, 0.30, 1.0

    active_background_color = 0.16, 0.22, 0.33, 1.0
    active_border_color     = 0.16, 0.29, 0.5, 1.0
    active_border_width     = 1

    hover_background_color  = 1.0, 1.0, 1.0, 0.1
    hover_border_color      = 1.0, 1.0, 1.0, 0.4
    hover_border_width      = 1

    preferences: "TEXTENSION_PG_suggestions"

    line_padding           = 1.25
    text_padding           = 5
    scrollbar_width        = 16
    fixed_font_size        = 16
    use_auto_font_size     = True
    use_bold_matches       = False
    foreground_color       = (0.4, 0.7, 1.0, 1.0)
    match_foreground_color = (0.87, 0.60, 0.25, 1.0)

    def __init__(self, st: bpy.types.SpaceTextEditor):
        super().__init__(parent=None)

        self.update_uniforms(shadow=(0, 0, 0, 0.5))
        self.description = Description(self)
        self.st = st

    @property
    def font_size(self):
        if self.use_auto_font_size:
            return self.st.font_size
        return self.fixed_font_size

    def poll(self) -> bool:
        if text := self.is_visible and _context.edit_text:
            if _get_sync_key(text) == self.sync_key:
                return bool(self.lines)
            # TODO: Setting this in the poll isn't a good idea.
            self.last_position = -1, -1
            self.sync_key = ()
        return False

    def sync_cursor(self, line_index) -> None:
        self.sync_key = _get_sync_key(_context.edit_text)
        self.last_position = line_index, self.sync_key[1]
        return None

    def draw(self) -> None:
        # Align the box below the cursor.
        st = _context.space_data
        x, y = st.region_location_from_cursor(*self.last_position)
        w, h = self.rect.size

        y -= h - st.offsets[1] - round(4 * _system.wu * 0.05)

        self.rect.draw(x, y, w, h)
        super().draw()  # ListBox.draw
        self.description.draw()

    def draw_entry(self, entry, x: int, y: int):
        length = entry.get_completion_prefix_length()
        string = entry.name

        if length == 0:
            self.draw_string(string, x, y)

        else:
            import blf
            prefix = string[:length]

            if self.use_bold_matches:
                blf.enable(1, BLF_BOLD)

            blf.position(1, x, y, 0)
            blf.color(1, *self.match_foreground_color)
            blf.draw(1, prefix)
            blf.disable(1, BLF_BOLD)

            x += blf.dimensions(1, prefix)[0]
            self.draw_string(string[length:], x, y)

    def dismiss(self):
        if self.is_visible:
            self.is_visible = False
            utils.safe_redraw()
            ui.idle_update()

    def hit_test(self, x, y):
        return self.description.hit_test(x, y) or super().hit_test(x, y)


@utils.factory
def get_instance() -> Suggestions:
    return utils.make_space_data_instancer(Suggestions)


def test_suggestions_box(x, y):
    instance = get_instance()
    if instance.poll():
        return instance.hit_test(x, y)
    return None


def draw_suggestions():
    """Draw callback for suggestions box."""
    instance = get_instance()
    if instance.poll():
        instance.draw()


@classmethod
def instance_poll(cls, context):
    if isinstance(context.space_data, bpy.types.SpaceTextEditor):
        return get_instance().poll()
    return False


class TEXTENSION_OT_suggestions_commit(utils.TextOperator):
    utils.km_def("Text Generic", 'RET', 'PRESS')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    poll = instance_poll

    def execute(self, context):
        instance = get_instance()

        index = instance.active.index

        text = context.edit_text
        line, col = text.cursor_focus

        # The selected completion.
        completion = instance.lines[index]

        # The name which jedi is being asked to complete.
        word_start = col - completion.get_completion_prefix_length()
        projected = text.lines[line].body[word_start:col] + completion.complete

        # The query + completion, including parameter/function suffixes.
        completion_string = completion.name_with_symbols

        # Generate a mousemove.
        ui.idle_update()

        # Complete only if it either adds to, or modifies the query.
        if completion.complete or projected != completion_string:
            if text.cursor_anchor != (line, col):
                text.cursor_anchor = line, col
            text.curc = word_start
            text.write(completion_string)
            text.cursor = line, col + len(completion.complete)
            # TODO: Aren't we supposed to use dismiss?
            instance.is_visible = False
            return {'FINISHED'}

        # The string completes nothing. If the commit was via return key, pass through the event.
        # TODO: Aren't we supposed to use dismiss?
        instance.is_visible = False
        return {'PASS_THROUGH'}


class TEXTENSION_OT_suggestions_dismiss(utils.TextOperator):
    utils.km_def("Text", 'ESC', 'PRESS')
    bl_options = {'INTERNAL'}

    def execute(self, context):
        instance = get_instance()
        if instance.is_visible:
            instance.dismiss()
            return {'CANCELLED'}
        # Pass the event.
        return {'PASS_THROUGH'}


# TODO: Make this generic and move to ui module.
# TODO: This requires implementing widget focus.
class TEXTENSION_OT_suggestions_navigate(utils.TextOperator):
    utils.km_def("Text Generic", 'UP_ARROW', 'PRESS', repeat=True, action='UP')
    utils.km_def("Text Generic", 'DOWN_ARROW', 'PRESS', repeat=True, action='DOWN')
    utils.km_def("Text Generic", 'PAGE_UP', 'PRESS', repeat=True, action='PAGE_UP')
    utils.km_def("Text Generic", 'PAGE_DOWN', 'PRESS', repeat=True, action='PAGE_DOWN')

    bl_options = {'INTERNAL'}

    action: bpy.props.EnumProperty(
        items=[(v, v, v) for v in ("DOWN", "UP", "PAGE_DOWN", "PAGE_UP")])

    poll = instance_poll

    def execute(self, context):
        instance = get_instance()
        index = instance.active.index

        # Navigating up/down with only one item passes the event on.
        if self.action in {'UP', 'DOWN'} and len(instance.lines) == 1:
            return {'PASS_THROUGH'}

        # Visible lines minus 1 is easier to track when we move by pages.
        visible_lines = int(instance.visible_lines) - 1
        dist = 1 if "DOWN" in self.action else -1

        if "PAGE" in self.action:
            dist *= visible_lines
            dist = max(0, min(index + dist, len(instance.lines) - 1))
        else:
            dist = (index + dist) % len(instance.lines)

        # Adjust the view if the new index is outside of it.
        if dist < instance.top:
            instance.top = dist

        elif dist > instance.top + visible_lines:
            # Make the bottom flush with the index.
            remainder = instance.visible_lines % 1.0
            instance.top = dist - visible_lines - remainder

        instance.active.set_index(dist)
        instance.reset_cache()
        return {'FINISHED'}


def dismiss():
    """Hides the suggestions box."""
    get_instance().dismiss()


def on_insert(line, column, fmt) -> None:
    """Hook for TEXTENSION_OT_insert"""
    if _context.edit_text.lines[line].body[column - 1] not in separators and fmt != b"#":
        deferred_complete()
    else:
        dismiss()


def on_delete(line, column, fmt) -> None:
    """Hook for TEXTENSION_OT_delete"""
    text = _context.edit_text
    leading_string = text.lines[line].body[:column]
    if not leading_string.strip() or leading_string[-1:] in separators | {"."}:
        dismiss()
    elif leading_string.lstrip():
        instance = get_instance()
        instance.sync_cursor(line)
        if instance.poll() and fmt != b"#":  # If visible, run completions again.
            deferred_complete()


def deferred_complete():
    def wrapper(ctx=_context.copy()):
        with _context.temp_override(**ctx):
            bpy.ops.textension.suggestions_complete('INVOKE_DEFAULT')
    utils.defer(wrapper)


def _disable_gc():
    gc.disable()
    # Do not allow gc to be enabled by jedi. We manage it ourselves.
    gc.enable = None.__init__
    utils.defer(_enable_gc)


def _enable_gc():
    Suggestions._temp_lines.clear()
    gc.enable = _gc_enable
    gc.enable()


class TEXTENSION_OT_suggestions_complete(utils.TextOperator):
    utils.km_def("Text Generic", 'SPACE', 'PRESS', ctrl=1)

    poll = utils.text_poll

    def invoke(self, context, event):
        if not runtime.loaded:
            _setup(force=True)

        text = context.edit_text
        instance = get_instance()

        # Line index is O(N) so avoid syncing cursor as much as possible.
        if instance.sync_key != _get_sync_key(text):
            instance.sync_cursor(text.select_end_line_index)

        from .patches.common import complete

        _disable_gc()

        try:
            ret = complete(text)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            raise e from None
        else:
            bpy.ret = ret  # For introspection

            # We don't want garbage collection to spuriously run until things
            # have been rendered.
            instance._temp_lines += [instance.lines]
            instance.lines = ret

            # TODO: Weak.
            instance.is_visible = bool(instance.lines)
            utils.safe_redraw()
            # XXX: Causes inconsistent repeated typing. Disable for now.
            # ui.idle_update()
        finally:
            return {'FINISHED'}


_instances: tuple[Suggestions] = get_instance.__kwdefaults__["cache"].values()


def reset_entries(self, context):
    for instance in _instances:
        instance.reset_cache()
    utils.redraw_editors()


@utils.factory
def _set_runtime_uniforms():
    def retself(obj): return obj

    def wrapper(path, attr, value):
        getter = attrgetter(path) if path else retself
        for obj in map(getter, _instances):
            obj.update_uniforms(**{attr: value})
        utils.redraw_editors()
    return wrapper


def update_uniform(path):
    if "." in path:
        path, attr = path.rsplit(".", 1)
        prop_name = f'{path.rsplit(".", 1)[-1]}_{attr}'
    else:
        attr = prop_name = path
        path = None

    def on_update(self, context, *, path=path, attr=attr, prop_name=prop_name):
        value = getattr(self, prop_name)
        _set_runtime_uniforms(path, attr, value)

    return on_update


def update_corner_radius(self, context):
    get = attrgetter("active", "hover", "scrollbar", "scrollbar.thumb")

    value = self.corner_radius
    Suggestions.corner_radius = value

    for instance in _instances:
        instance.update_uniforms(corner_radius=value)
        for widget in get(instance):
            widget.update_uniforms(corner_radius=value)


def update_resize_highlights(self, context):
    ui.EdgeResizer.show_resize_handles = getattr(self, "show_resize_handles")
    get_sizers = attrgetter("resizer.sizers")
    zero_alpha = methodcaller("set_alpha", 0.0)
    utils.consume(map(zero_alpha, utils.starchain(map(get_sizers, _instances))))


def update_resizer_colors(self: "TEXTENSION_PG_suggestions", context):
    get_sizer_rects = attrgetter("resizer.horz.rect", "resizer.vert.rect")
    color = tuple(self.resizer_color) + (0.0,)
    for rect in utils.starchain(map(get_sizer_rects, _instances)):
        rect.background_color = color
        rect.border_color = color


def update_value(name: str):
    def update_setting(self: "TEXTENSION_PG_suggestions", context) -> None:
        setattr(Suggestions, name, getattr(self, name))
        for instance in _instances:
            instance.reset_cache()
        utils.redraw_editors('TEXT_EDITOR', 'WINDOW')
    return update_setting


def uniform_color_property(path):
    name = path.replace(".", "_")

    ret = bpy.props.FloatVectorProperty(
        name=name.replace("_", " ").title(),
        default=getattr(Suggestions, name),
        update=update_uniform(path),
        min=0.0,
        max=1.0,
        size=4,
        subtype='COLOR_GAMMA'
    )
    sys._getframe(1).f_locals["__annotations__"][name] = ret


def update_uniform_child(path: str, uniform: str):
    name = f"{path}_{uniform}".replace(".", "_")
    def update(self, context, *, get_child=attrgetter(path), name=name):
        value = getattr(self, name)
        setattr(Suggestions, name, value)
        for instance in _instances:
            get_child(instance).update_uniforms(**{uniform: value})
        utils.redraw_editors('TEXT_EDITOR', 'WINDOW')
    return update


def update_uniform(self, context):
    names = ("background_color", "border_color", "border_width")
    values = {name: getattr(self, name) for name in names}
    for instance in _instances:
        instance.update_uniforms(**values)
    for name in names:
        setattr(Suggestions, name, values[name])


class TEXTENSION_PG_suggestions(bpy.types.PropertyGroup):
    color_default_kw = {"min": 0, "max": 1, "size": 4, "subtype": 'COLOR_GAMMA'}

    fixed_font_size: bpy.props.IntProperty(
        update=update_value("fixed_font_size"),
        default=Suggestions.fixed_font_size,
        name="Font Size",
        max=144,
        min=1,
    )
    use_bold_matches: bpy.props.BoolProperty(
        update=update_value("use_bold_matches"),
        default=Suggestions.use_bold_matches,
        name="Bold Partial Matches",
    )
    use_auto_font_size: bpy.props.BoolProperty(
        update=update_value("use_auto_font_size"),
        default=Suggestions.use_auto_font_size,
        name="Automatic Font Size",
    )
    line_padding: bpy.props.FloatProperty(
        update=update_value("line_padding"),
        default=Suggestions.line_padding,
        name="Line Height",
        min=0.5,
        max=4.0,
    )
    text_padding: bpy.props.IntProperty(
        update=update_value("text_padding"),
        default=Suggestions.text_padding,
        name="Text Padding",
        max=1000,
        min=0,
    )
    scrollbar_width: bpy.props.IntProperty(
        name="Scrollbar Width",
        update=update_value("scrollbar_width"),
        default=Suggestions.scrollbar_width,
        max=100,
        min=0,
    )
    foreground_color: bpy.props.FloatVectorProperty(
        update=update_value("foreground_color"),
        default=Suggestions.foreground_color,
        name="Foreground Color",
        **color_default_kw,
    )
    match_foreground_color: bpy.props.FloatVectorProperty(
        update=update_value("match_foreground_color"),
        default=Suggestions.match_foreground_color,
        name="Match Foreground Color",
        **color_default_kw,
    )
    corner_radius: bpy.props.FloatProperty(
        update=update_corner_radius,
        name="Roundness",
        default=0.0,
        max=10.0,
        min=0.0,
    )

    show_resize_handles: bpy.props.BoolProperty(
        update=update_resize_highlights,
        name="Show Resize Highlights",
        default=True
    )
    resizer_color: bpy.props.FloatVectorProperty(
        default=(0.38, 0.38, 0.38), **(color_default_kw | {"size": 3}),
        update=update_resizer_colors,
        name="Resize Handle Color"        
    )

    background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.background_color,
        update=update_uniform,
        **color_default_kw
    )
    border_color: bpy.props.FloatVectorProperty(
        default=Suggestions.border_color,
        update=update_uniform,
        **color_default_kw
    )
    border_width: bpy.props.FloatProperty(
        default=Suggestions.border_width,
        update=update_uniform,
        min=0.0,
        max=20.0
    )

    active_background_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("active", "background_color"),
        default=Suggestions.active_background_color,
        **color_default_kw
    )
    active_border_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("active", "border_color"),
        default=Suggestions.active_border_color,
        **color_default_kw,
    )
    active_border_width: bpy.props.FloatProperty(
        update=update_uniform_child("active", "border_width"),
        default=Suggestions.active_border_width,
        max=20.0,
        min=0.0
    )

    hover_background_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("hover", "background_color"),
        default=Suggestions.hover_background_color,
        **color_default_kw,
    )
    hover_border_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("hover", "border_color"),
        default=Suggestions.hover_border_color,
        **color_default_kw
    )
    hover_border_width: bpy.props.FloatProperty(
        update=update_uniform_child("hover", "border_width"),
        default=Suggestions.hover_border_width,
        max=20.0,
        min=0.0
    )

    scrollbar_background_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("scrollbar", "background_color"),
        default=Suggestions.scrollbar_background_color,
        **color_default_kw
    )
    scrollbar_border_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("scrollbar", "border_color"),
        default=Suggestions.scrollbar_border_color,
        **color_default_kw
    )
    scrollbar_border_width: bpy.props.FloatProperty(
        update=update_uniform_child("scrollbar", "border_width"),
        default=Suggestions.scrollbar_border_width,
        max=20.0,
        min=0.0
    )

    scrollbar_thumb_background_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("scrollbar.thumb", "background_color"),
        default=Suggestions.scrollbar_thumb_background_color,
        **color_default_kw
    )
    scrollbar_thumb_border_color: bpy.props.FloatVectorProperty(
        update=update_uniform_child("scrollbar.thumb", "border_color"),
        default=Suggestions.scrollbar_thumb_border_color,
        **color_default_kw
    )
    scrollbar_thumb_border_width: bpy.props.FloatProperty(
        update=update_uniform_child("scrollbar.thumb", "border_width"),
        default=Suggestions.scrollbar_thumb_border_width,
        max=20.0,
        min=0.0
    )


classes = (
    TEXTENSION_PG_suggestions,

    TEXTENSION_OT_suggestions_complete,
    TEXTENSION_OT_suggestions_navigate,
    TEXTENSION_OT_suggestions_dismiss,
    TEXTENSION_OT_suggestions_commit,
)


def draw_settings(prefs, context, layout):
    suggestions = prefs.suggestions
    layout.prop(suggestions, "use_auto_font_size")
    row = layout.row()
    row.prop(suggestions, "fixed_font_size")
    row.enabled = not suggestions.use_auto_font_size

    layout.prop(suggestions, "use_bold_matches")
    layout.prop(suggestions, "show_resize_handles")
    layout.prop(suggestions, "line_padding")
    layout.prop(suggestions, "text_padding")
    layout.separator(factor=3)
    layout.prop(suggestions, "corner_radius", slider=True)
    layout.prop(suggestions, "scrollbar_width")
    layout.separator(factor=3)
    layout.prop(suggestions, "foreground_color")
    layout.prop(suggestions, "match_foreground_color")
    layout.prop(suggestions, "background_color")
    layout.prop(suggestions, "border_color")
    layout.prop(suggestions, "border_width")
    layout.separator(factor=3)

    layout.prop(suggestions, "active_background_color")
    layout.prop(suggestions, "active_border_color")
    layout.prop(suggestions, "active_border_width")
    layout.separator(factor=3)
    layout.prop(suggestions, "hover_background_color")
    layout.prop(suggestions, "hover_border_color")
    layout.prop(suggestions, "hover_border_width")
    layout.separator(factor=3)
    layout.prop(suggestions, "scrollbar_thumb_background_color")
    layout.prop(suggestions, "scrollbar_thumb_border_color")
    layout.prop(suggestions, "scrollbar_thumb_border_width")
    layout.separator(factor=3)
    layout.prop(suggestions, "resizer_color")
    layout.separator(factor=3)
    layout.prop(suggestions, "scrollbar_background_color")
    layout.prop(suggestions, "scrollbar_border_color")
    layout.prop(suggestions, "scrollbar_border_width")


def apply_custom_settings():
    p = Suggestions.preferences
    for name in (
        "background_color",
        "border_color",
        "border_width",
        "corner_radius",

        "active_background_color",
        "active_border_color",
        "active_border_width",

        "hover_background_color",
        "hover_border_color",
        "hover_border_width",
        
        "scrollbar_background_color",
        "scrollbar_border_color",
        "scrollbar_border_width",

        "scrollbar_thumb_background_color",
        "scrollbar_thumb_border_color",
        "scrollbar_thumb_border_width"
    ):
        setattr(Suggestions, name, getattr(p, name))


def _setup(force=False):
    if runtime.loaded:
        return

    if force and bpy.app.timers.is_registered(_setup):
        bpy.app.timers.unregister(_setup)

    import jedi
    from . import patches

    # Do not let jedi infer anonymous parameters. It's slow and useless.
    jedi.settings.dynamic_params = False
    jedi.settings.auto_import_modules = set()

    patches.apply()
    runtime.loaded = True


def enable():
    # Unless Jedi already exists, it's placed into the directory 'download'.
    # In this case we add it to sys.path to make it globally importable.
    plugin_path = os.path.dirname(__file__)
    if plugin_path not in sys.path:  # TODO: Should be 'download', not root directory.
        sys.path.append(plugin_path)

    utils.register_classes(classes)

    from textension import ui, prefs
    from textension.overrides import default
    default.insert_hooks.append(on_insert)
    default.delete_hooks.append(on_delete)

    Suggestions.preferences = prefs.add_settings(TEXTENSION_PG_suggestions)
    apply_custom_settings()
    utils.add_draw_hook(draw_suggestions)
    ui.add_hit_test(test_suggestions_box)

    # Override the default auto complete operator.
    TEXT_OT_autocomplete.apply_override()

    # Defer loading jedi and applying patches so the plugin is enabled faster.
    bpy.app.timers.register(_setup, first_interval=0.3)


def disable():
    from textension import ui, prefs

    from textension.overrides import default
    default.insert_hooks.remove(on_insert)
    default.delete_hooks.remove(on_delete)
    TEXT_OT_autocomplete.remove_override()

    prefs.remove_settings(TEXTENSION_PG_suggestions)
    ui.remove_hit_test(test_suggestions_box)
    utils.remove_draw_hook(draw_suggestions)
    utils.unregister_classes(classes)
    get_instance.__kwdefaults__["cache"].clear()
