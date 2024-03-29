"""This module implements auto-completions."""


from textension.btypes.defs import OPERATOR_CANCELLED, BLF_BOLD
from textension.overrides import OpOverride
from textension.utils import _context, _system, get_text_line_sync_key
from textension import utils, ui, prefs

import operator

import sys
import bpy
import gc

from . import hover


settings = utils.namespace(
    # Plugin dependencies are loaded externally via a timer.
    loaded=False,

    use_fuzzy_search          = True,
    use_ordered_fuzzy_search  = True,
    use_case_sensitive_search = False,

    # Whether the hover module is loaded. Regardless of preferences.
    show_hover                = None,
)


# Backup.
_gc_enable = gc.enable

# Token separators excluding dot/period
separators = {*" !\"#$%&\'()*+,-/:;<=>?@[\\]^`{|}~"}


class TEXT_OT_autocomplete(OpOverride):
    def invoke(self):
        deferred_complete()
        return OPERATOR_CANCELLED


def _get_extended_api_type(name):
    from .patches.common import get_extended_type

    return get_extended_type(name)


class Description(ui.widgets.TextView):
    parent: "Suggestions"
    last_entry       = None

    foreground_color = utils._forwarder("parent.description_foreground_color")
    background_color = utils._forwarder("parent.description_background_color")
    font_size        = utils._forwarder("parent.description_font_size")
    line_padding     = utils._forwarder("parent.description_line_padding")
    use_word_wrap    = utils._forwarder("parent.description_use_word_wrap")
    shadow           = utils._forwarder("parent.shadow")

    @property
    def active_entry(self):
        if self.parent.active.index != -1:
            return self.parent.lines[self.parent.active.index]
        return None

    def draw(self):
        entry = self.active_entry

        if entry is not self.last_entry:
            if entry:
                api_type = _get_extended_api_type(entry._name)
                string = f"type: {api_type}\n\n"
                string += entry.docstring(fast=False)
            else:
                string = ()
            self.set_from_string(string)
            self.last_entry = entry
        return super().draw()

    def add_font_delta(self, delta: int):
        parent = self.parent

        rel_size = parent.description_relative_font_size + delta
        abs_size = parent.space_data.font_size + rel_size

        rel_size -= abs_size - max(5, abs_size)
        Suggestions.description_relative_font_size = rel_size
        self._update_lines()
        utils.safe_redraw()

    @property
    def position(self):
        # Position the box so it's flush with Suggestion's top edge.
        x, y = self.parent.rect.position + self.parent.rect.size
        return x, y - self.height


class Suggestions(ui.widgets.ListBox):
    default_width  = 260
    default_height = 158

    preferences: "TEXTENSION_PG_suggestions"
    height_hint: int

    font_id                 = 1
    _temp_lines             = []
    last_position           = 0, 0
    sync_key                = ()
    last_nlines             = 0
    is_visible              = False

    show_description          = False
    show_bold_matches         = False

    relative_font_size        = -2
    use_fuzzy_search          = True
    use_ordered_fuzzy_search  = True
    use_case_sensitive_search = False

    shadow                  = 0.0,  0.0,  0.0,  0.5
    background_color        = 0.15, 0.15, 0.15, 1.0
    border_color            = 0.30, 0.30, 0.30, 1.0

    active_background_color = 0.16, 0.22, 0.33, 1.0
    active_border_color     = 0.16, 0.29, 0.50, 1.0
    active_border_width     = 1

    hover_background_color  = 1.0, 1.0, 1.0, 0.1
    hover_border_color      = 1.0, 1.0, 1.0, 0.4
    hover_border_width      = 1

    foreground_color        = 0.4,  0.7,  1.0,  1.0
    match_foreground_color  = 0.87, 0.60, 0.25, 1.0

    description_foreground_color = 0.7,  0.7,  0.7,  1.0
    description_background_color = 0.18, 0.18, 0.18, 1.0
    description_line_padding     = 1.25
    description_use_monospace    = False
    description_use_word_wrap    = True
    description_relative_font_size = -2

    def __init__(self, st):
        super().__init__(parent=None)
        self.space_data = st
        self.description = Description(self)
        self.height_hint = self.height
        update_defaults()

    def resize(self, size):
        super().resize(size)
        self.height_hint = size[1]

    @property
    def font_size(self):
        return max(1, int((self.space_data.font_size + self.relative_font_size) * ui.runtime.wu_norm))

    @property
    def description_font_size(self):
        return max(1, int((self.space_data.font_size + self.description_relative_font_size) * ui.runtime.wu_norm))

    def poll(self) -> bool:
        if text := self.is_visible and _context.edit_text:
            if get_text_line_sync_key(text) == self.sync_key:
                return bool(self.lines)
            # TODO: Setting this in the poll isn't a good idea.
            self.last_position = -1, -1
            self.sync_key = ()
            self.is_visible = False
        return False

    def sync_cursor(self, line_index) -> None:
        self.sync_key = get_text_line_sync_key(_context.edit_text)
        self.last_position = line_index, self.sync_key[1]
        return None

    @property
    def position(self):
        # Align the box below the cursor.
        st = _context.space_data
        x, y = st.region_location_from_cursor(*self.last_position)
        y -= self.rect.height - st.offsets.y - round(4 * _system.wu * 0.05)
        return x, y

    def draw(self) -> None:
        super().draw()
        if self.show_description:
            self.description.draw()

    def draw_entry(self, entry, x: int, y: int):
        length = entry.get_completion_prefix_length()
        string = entry.name

        if length == 0:
            self.draw_string(string, x, y)

        else:
            # TODO: This could benefit from cleanup.
            import blf
            like_name = entry.test_like_name

            if settings.use_case_sensitive_search:
                like_name = entry._like_name
                test_string = string

            else:
                test_string = string.lower()

            draw_list = []

            # If the full like name is in the string, use that.
            if like_name in test_string:
                span = len(like_name)
                start = test_string.index(like_name)

                if start is not 0:
                    draw_list += (string[:start], False),

                end = start + span
                draw_list += (string[start:end], True),

                if end != len(string):
                    draw_list += (string[end:], False),

            else:

                from itertools import repeat
                draw_list = list(zip(string, repeat(False)))

                if settings.use_ordered_fuzzy_search:
                    index = -1
                    for c in like_name:
                        index = test_string.index(c, index + 1)
                        draw_list[index] = (draw_list[index][0], True)

                else:
                    for c in like_name:
                        index = test_string.index(c)
                        draw_list[index] = (draw_list[index][0], True)
                        test_string = test_string.replace(c, "\x00", 1)

            for char, is_match in draw_list:
                if is_match:
                    if self.show_bold_matches:
                        blf.enable(self.font_id, BLF_BOLD)
                    color = self.match_foreground_color
                else:
                    color = self.foreground_color

                blf.color(self.font_id, *color)

                blf.position(self.font_id, x, y, 0)
                blf.draw(self.font_id, char)
                blf.disable(self.font_id, BLF_BOLD)

                x += blf.dimensions(self.font_id, char)[0]

    def dismiss(self, update_cursor=True):
        if self.is_visible:
            self.is_visible = False
            utils.safe_redraw_from_space(self.space_data)
            if update_cursor:
                ui.idle_update()

    @utils.set_name("hit_test (Suggestions)")
    def hit_test(self, x, y):
        return self.description.hit_test(x, y) or super().hit_test(x, y)

    def toggle_description(self, visible: bool = None):
        if visible is None:
            visible = not self.show_description
        self.show_description = visible
        utils.safe_redraw()

    def on_activate(self) -> None:
        self.active.index = self.hover.index
        bpy.ops.textension.suggestions_commit('INVOKE_DEFAULT')


@utils.inline
def get_instance() -> Suggestions:
    return utils.make_space_data_instancer(Suggestions)


def hit_test_suggestions(x, y):
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


class TEXTENSION_OT_suggestions_complete(utils.TextOperator):
    toggle_description: bpy.props.BoolProperty(default=True)

    poll = utils.text_poll

    def execute(self, context):
        instance = get_instance()

        if instance.is_visible and self.toggle_description:
            instance.toggle_description()
            return {'FINISHED'}

        if not settings.loaded:
            _setup_jedi(force=True)

        text = context.edit_text
        nlines = context.space_data.drawcache.total_lines

        # Text line index is O(N) so avoid syncing cursor as much as possible.
        if instance.sync_key != get_text_line_sync_key(text) or instance.last_nlines != nlines:
            instance.sync_cursor(text.select_end_line_index)
            instance.last_nlines = nlines

        from .patches.common import complete

        _disable_gc()

        try:
            ret = complete(text)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            print("Error at:", tuple(text.cursor_focus))
            raise e from None
        else:
            # We don't want garbage collection to spuriously run until things
            # have been rendered.
            instance._temp_lines += instance.lines,
            instance.lines = ret

            height = instance.content_height
            if instance.lines and height <= instance.rect.height_inner:
                instance.rect.height_inner = height
            else:
                instance.rect.height = instance.height_hint

            # TODO: Weak.
            instance.is_visible = bool(instance.lines)
            utils.safe_redraw()
            # XXX: Causes inconsistent repeated typing. Disable for now.
            # ui.idle_update()
        finally:
            return {'FINISHED'}


class TEXTENSION_OT_suggestions_commit(utils.TextOperator):
    utils.km_def("Text Generic", 'RET', 'PRESS')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    poll = instance_poll

    def execute(self, context):
        instance = get_instance()
        entry = instance.active_entry

        if not entry:
            return {'CANCELLED'}

        text = context.edit_text
        line, column = text.cursor_focus

        # The name which jedi is being asked to complete.
        word_start = column - entry._like_name_length
        query = text.lines[line].body[word_start:column]

        if settings.use_fuzzy_search:
            word = entry.string_name

        else:
            word = query + entry.complete

        # Same as ``word`` except it may append ``=`` for parameters.
        completion_string = entry.name_with_symbols

        instance.dismiss()
        if word == query == completion_string:
            # The string completes nothing.
            return {'PASS_THROUGH'}

        text.cursor_anchor = line, column
        text.curc = word_start
        text.write(completion_string)
        text.cursor = line, column + len(completion_string) - len(query)
        return {'FINISHED'}



class TEXTENSION_OT_suggestions_dismiss(utils.TextOperator):
    utils.km_def("Text", 'ESC', 'PRESS')
    bl_options = {'INTERNAL'}

    def execute(self, context):
        instance = get_instance()
        if instance.is_visible:
            instance.dismiss()
            return {'CANCELLED'}

        else:
            # Not visible, give the event to other operators.
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
        top = instance.top

        if "PAGE" in self.action:
            dist *= visible_lines
            dist = max(0, min(index + dist, len(instance.lines) - 1))
        else:
            dist = (index + dist) % len(instance.lines)

        instance.active.set_index(dist)
        # Adjust the view if the new index is outside of it.
        if dist < top:
            instance.top = dist

        elif dist > top + visible_lines:
            # Make the bottom flush with the index.
            remainder = instance.visible_lines % 1.0
            instance.top = dist - visible_lines - remainder

        if top != instance.top:
            instance.reset_cache()
        return {'FINISHED'}


def dismiss():
    """Hides the suggestions box."""
    get_instance().dismiss()


def on_insert(line, column, fmt) -> None:
    """Hook for TEXTENSION_OT_insert"""
    if _context.edit_text.lines[line].body[column - 1] not in separators and fmt != b"#":
        deferred_complete(toggle_description=False)
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
            deferred_complete(toggle_description=False)


def deferred_complete(toggle_description=True):
    def wrapper(ctx=_context.copy()):
        with utils.context_override(**ctx):
            bpy.ops.textension.suggestions_complete(toggle_description=toggle_description)
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


def update_defaults(self: "TEXTENSION_PG_suggestions" = None, context = None):
    p = Suggestions.preferences

    new_settings = {}

    for key in p.bl_rna.properties.keys():
        if hasattr(Suggestions, key):
            value = p.path_resolve(key)

            # Uniform blocks don't understand bpy prop array sequences.
            if isinstance(value, utils.bpy_array_types):
                value = tuple(value)
            new_settings[key] = value

    # Update defaults on the class.
    utils._update_namespace(Suggestions, **new_settings)

    Description.font_id = int(Suggestions.description_use_monospace)

    # ``corner_radius`` is omitted since it's global.
    names = ("background_color", "border_color", "border_width")

    ui.widgets.EdgeResizer.show_resize_handles = p.show_resize_handles
    sizer_color = tuple(p.resizer_color) + (0.0,)
    sizer_uniforms = dict.fromkeys(names[:2], sizer_color)

    def update_scrollbar(cls, scrollbar):
        scrollbar.update_uniforms(
            background_color=cls.scrollbar_background_color,
            border_color=cls.scrollbar_border_color,
            border_width=cls.scrollbar_border_width,
        )
        scrollbar.thumb.update_uniforms(
            background_color=cls.scrollbar_thumb_background_color,
            border_color=cls.scrollbar_thumb_border_color,
            border_width=cls.scrollbar_thumb_border_width,
        )

    for instance in Suggestions.instances:
        instance.set_corner_radius(Suggestions.corner_radius)
        instance.update_from_defaults()
        instance.description.update_from_defaults()
        instance.description._update_lines()
        instance.description.set_corner_radius(Suggestions.corner_radius)

        # Get the corresponding uniform value stored on the class.
        for name in ("active", "hover"):
            concat = map((name.replace(".", "_") + "_").__add__, names)

            widget = operator.attrgetter(name)(instance)
            uniforms = dict(zip(names, map(new_settings.__getitem__, concat)))
            widget.update_uniforms(corner_radius=Suggestions.corner_radius,
                                   **uniforms)

        Description.scrollbar_width = p.scrollbar_width
        update_scrollbar(Suggestions, instance.scrollbar)
        update_scrollbar(Suggestions, instance.description.scrollbar)
        update_scrollbar(Suggestions, instance.description.scrollbar_h)

        # Resizer settings.
        for sizer in instance.resizer.sizers:
            sizer.update_uniforms(**sizer_uniforms)
            sizer.set_alpha(0.0)

        instance.reset_cache()

    runtime_names = ("use_fuzzy_search", "use_ordered_fuzzy_search", "use_case_sensitive_search")
    curr_runtime = tuple(map(settings.__getattribute__, runtime_names))

    values = map(p.path_resolve, runtime_names)
    settings.update(**dict(zip(runtime_names, values)))

    if curr_runtime != tuple(map(settings.__getattribute__, runtime_names)):
        instance_cache = get_instance.__kwdefaults__["cache"]

        # If an instance is already open, complete with new settings.
        for window in _context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != 'TEXT_EDITOR':
                    continue
                st = area.spaces.active
                instance = instance_cache.get(st)
                if not instance or not instance.is_visible:
                    continue
                with utils.context_override(space_data=st, window=window, area=area):
                    bpy.ops.textension.suggestions_complete(toggle_description=False)

    update_hover_settings(p, None)


def update_hover_settings(self: "TEXTENSION_PG_suggestions", context):
    if self.show_hover:
        if not settings.show_hover:
            hover.enable()

    elif settings.show_hover:
        hover.disable()

    settings.show_hover = self.show_hover


class TEXTENSION_PG_suggestions(bpy.types.PropertyGroup):
    runtime = utils.namespace(
        show_general_settings=True,
        show_theme_settings=False,
        show_description_settings=False,

        show_theme_listbox=False,
        show_theme_entry=False,
        show_theme_scrollbar=False,

        show_hover_settings=False,
    )
    color_default_kw = {"min": 0, "max": 1, "size": 4, "subtype": 'COLOR_GAMMA'}

    use_case_sensitive_search: bpy.props.BoolProperty(
        description="Suggestions are case-sensitive to the typed text",
        default=Suggestions.use_case_sensitive_search,
        update=update_defaults,
    )
    use_fuzzy_search: bpy.props.BoolProperty(
        description="Show completions with partial (fuzzy) matches",
        default=Suggestions.use_fuzzy_search,
        update=update_defaults,
    )
    use_ordered_fuzzy_search: bpy.props.BoolProperty(
        description="Use strict fuzzy search order. \n"
                    "When enabled, `pain` won't match `pineapple`",
        default=Suggestions.use_ordered_fuzzy_search,
        update=update_defaults,
    )
    show_bold_matches: bpy.props.BoolProperty(
        description="Show the matching part of a completion in bold",
        default=Suggestions.show_bold_matches,
        update=update_defaults,
    )
    show_resize_handles: bpy.props.BoolProperty(
        description="Highlight resize handles when hovered",
        update=update_defaults,
        default=True
    )
    relative_font_size: bpy.props.IntProperty(
        description="Relative font size when automatic font size is enabled",
        default=Suggestions.relative_font_size,
        min=-144,
        max=144,
        update=update_defaults
    )
    line_padding: bpy.props.FloatProperty(
        default=Suggestions.line_padding,
        update=update_defaults,
        min=0.5,
        max=4.0,
    )
    text_padding: bpy.props.IntProperty(
        default=Suggestions.text_padding,
        update=update_defaults,
        max=1000,
        min=0,
    )
    scrollbar_width: bpy.props.IntProperty(
        default=Suggestions.scrollbar_width,
        update=update_defaults,
        max=100,
        min=0,
    )
    
    shadow: bpy.props.FloatVectorProperty(
        default=Suggestions.shadow,
        update=update_defaults,
        **color_default_kw
    )

    foreground_color: bpy.props.FloatVectorProperty(
        default=Suggestions.foreground_color,
        update=update_defaults,
        **color_default_kw,
    )
    match_foreground_color: bpy.props.FloatVectorProperty(
        default=Suggestions.match_foreground_color,
        update=update_defaults,
        **color_default_kw,
    )
    corner_radius: bpy.props.FloatProperty(
        update=update_defaults,
        default=0.0,
        max=10.0,
        min=0.0,
    )

    resizer_color: bpy.props.FloatVectorProperty(
        default=(0.38, 0.38, 0.38),
        update=update_defaults,
        **(color_default_kw | {"size": 3}),
    )

    background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.background_color,
        update=update_defaults,
        **color_default_kw
    )
    border_color: bpy.props.FloatVectorProperty(
        default=Suggestions.border_color,
        update=update_defaults,
        **color_default_kw
    )
    border_width: bpy.props.IntProperty(
        default=int(Suggestions.border_width),
        update=update_defaults,
        min=0,
        max=20
    )

    active_background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.active_background_color,
        update=update_defaults,
        **color_default_kw
    )
    active_border_color: bpy.props.FloatVectorProperty(
        default=Suggestions.active_border_color,
        update=update_defaults,
        **color_default_kw,
    )
    active_border_width: bpy.props.FloatProperty(
        default=Suggestions.active_border_width,
        update=update_defaults,
        max=20.0,
        min=0.0
    )

    hover_background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.hover_background_color,
        update=update_defaults,
        **color_default_kw,
    )
    hover_border_color: bpy.props.FloatVectorProperty(
        default=Suggestions.hover_border_color,
        update=update_defaults,
        **color_default_kw
    )
    hover_border_width: bpy.props.FloatProperty(
        default=Suggestions.hover_border_width,
        update=update_defaults,
        max=20.0,
        min=0.0
    )

    scrollbar_background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.scrollbar_background_color,
        update=update_defaults,
        **color_default_kw
    )
    scrollbar_border_color: bpy.props.FloatVectorProperty(
        default=Suggestions.scrollbar_border_color,
        update=update_defaults,
        **color_default_kw
    )
    scrollbar_border_width: bpy.props.FloatProperty(
        default=Suggestions.scrollbar_border_width,
        update=update_defaults,
        max=20.0,
        min=0.0
    )

    scrollbar_thumb_background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.scrollbar_thumb_background_color,
        update=update_defaults,
        **color_default_kw
    )
    scrollbar_thumb_border_color: bpy.props.FloatVectorProperty(
        default=Suggestions.scrollbar_thumb_border_color,
        update=update_defaults,
        **color_default_kw
    )
    scrollbar_thumb_border_width: bpy.props.FloatProperty(
        default=Suggestions.scrollbar_thumb_border_width,
        update=update_defaults,
        max=20.0,
        min=0.0
    )

    description_use_word_wrap: bpy.props.BoolProperty(
        update=update_defaults,
        default=True
    )
    description_relative_font_size: bpy.props.IntProperty(
        description="Relative font size when automatic font size is enabled",
        default=Suggestions.description_relative_font_size,
        min=-144,
        max=144,
        update=update_defaults
    )
    description_line_padding: bpy.props.FloatProperty(
        default=Suggestions.description_line_padding,
        update=update_defaults,
        min=0.5,
        max=4.0,
    )
    description_foreground_color: bpy.props.FloatVectorProperty(
        default=Suggestions.description_foreground_color,
        update=update_defaults,
        **color_default_kw,
    )
    description_background_color: bpy.props.FloatVectorProperty(
        default=Suggestions.description_background_color,
        update=update_defaults,
        **color_default_kw,
    )
    description_use_monospace: bpy.props.BoolProperty(
        default=Suggestions.description_use_monospace,
        update=update_defaults
    )
    show_hover: bpy.props.BoolProperty(
        description="Show hover information",
        default=True,
        update=update_hover_settings
    )
    hover_delay_ms: bpy.props.IntProperty(
        default=300,
        min=0,
        max=5000,
        description="Delay in milliseconds before showing hover"
    )


classes = (
    TEXTENSION_PG_suggestions,

    TEXTENSION_OT_suggestions_complete,
    TEXTENSION_OT_suggestions_navigate,
    TEXTENSION_OT_suggestions_dismiss,
    TEXTENSION_OT_suggestions_commit,
)


def add_runtime_toggle(layout, path, text, emboss=True):
    path = "suggestions.runtime." + path
    c = layout.column(align=True)
    c.emboss = 'NORMAL'
    r = c.row()
    value = prefs.resolve_prefs_path(path)
    if not emboss:
        r2 = r.row()
        r2.alignment = 'LEFT'
        op = r2.operator("textension.ui_show",
                        text="",
                        depress=value,
                        emboss=emboss,
                        icon='TRIA_DOWN' if value else 'TRIA_RIGHT')
        op.path = path
    op = r.operator("textension.ui_show",
                    text=text,
                    depress=value,
                    emboss=emboss,
                    icon=('TRIA_DOWN' if value else 'TRIA_RIGHT') if emboss else 'NONE')

    op.path = path
    r.alignment = 'EXPAND'# if emboss else 'LEFT'
    if value:
        c.separator()
    return value and c


def draw_settings(prefs, context, layout):

    suggestions = prefs.suggestions

    layout = layout.column(align=True)
    layout.use_property_split = True
    layout.use_property_decorate = False

    # General settings.
    if c := add_runtime_toggle(layout, "show_general_settings", "General"):
        c.prop(suggestions, "use_case_sensitive_search", text="Match Case")
        c.prop(suggestions, "use_fuzzy_search", text="Fuzzy Search")

        r = c.row()
        r.prop(suggestions, "use_ordered_fuzzy_search", text="Ordered Search")
        r.enabled = suggestions.use_fuzzy_search

        c.prop(suggestions, "show_bold_matches", text="Bold Matches")
        c.prop(suggestions, "show_resize_handles", text="Highlight Resizers")
        c.separator()

        c.prop(suggestions, "line_padding", text="Line Height")
        c.prop(suggestions, "relative_font_size", text="Relative Font Size")

        c.separator(factor=3)
        c.prop(suggestions, "corner_radius", slider=True, text="Roundness")
        c.prop(suggestions, "scrollbar_width", text="Scrollbar Width")
        c.separator(factor=3)

    # Theme settings.
    if c := add_runtime_toggle(layout, "show_theme_settings", "Theme"):
        c.prop(suggestions, "shadow", text="Shadow Color")

        # ListBox Theme settings.
        if p := add_runtime_toggle(c, "show_theme_listbox", "List Box", emboss=False):
            p.prop(suggestions, "foreground_color", text="Foreground")
            p.prop(suggestions, "background_color", text="Background")
            p.prop(suggestions, "match_foreground_color", text="Match Color")
            p.prop(suggestions, "border_color", text="Border")
            p.prop(suggestions, "border_width", text="Border Width")
            p.prop(suggestions, "resizer_color", text="Resize Handle")
            p.separator(factor=1)

        # Entry Theme settings.
        if p := add_runtime_toggle(c, "show_theme_entry", "Entry", emboss=False):
            c.prop(suggestions, "active_background_color", text="Active Background")
            c.prop(suggestions, "active_border_color", text="Active Border Color")
            c.prop(suggestions, "active_border_width", text="Active Border Width")
            c.separator(factor=1)
            c.prop(suggestions, "hover_background_color", text="Hover Background")
            c.prop(suggestions, "hover_border_color", text="Hover Border Color")
            c.prop(suggestions, "hover_border_width", text="Hover Border Width")
            c.separator(factor=1)

        # Scrollbar Theme settings.
        if p := add_runtime_toggle(c, "show_theme_scrollbar", "Scrollbar", emboss=False):
            c.prop(suggestions, "scrollbar_background_color", text="Background")
            c.prop(suggestions, "scrollbar_border_color", text="Border Color")
            c.prop(suggestions, "scrollbar_border_width", text="Border Width")
            c.separator(factor=1)
            c.prop(suggestions, "scrollbar_thumb_background_color", text="Thumb Background")
            c.prop(suggestions, "scrollbar_thumb_border_color", text="Thumb Border Color")
            c.prop(suggestions, "scrollbar_thumb_border_width", text="Thumb Border Width")

        c.separator()

    # Description settings.
    if c := add_runtime_toggle(layout, "show_description_settings", "Description"):
        c.prop(suggestions, "description_use_word_wrap", text="Use Word Wrap")
        c.prop(suggestions, "description_use_monospace", text="Use Monospace")

        r = c.row()
        r.prop(suggestions, "description_relative_font_size", text="Relative Font Size")

        c.prop(suggestions, "description_line_padding", text="Line Height")
        c.prop(suggestions, "description_foreground_color", text="Foreground")
        c.prop(suggestions, "description_background_color", text="Background")

    if c := add_runtime_toggle(layout, "show_hover_settings", "Hover"):
        c.prop(suggestions, "show_hover", text="Show Hover")
        c.prop(suggestions, "hover_delay_ms", text="Delay")


def _setup_jedi(force=False):
    if settings.loaded:
        return

    if force and bpy.app.timers.is_registered(_setup_jedi):
        bpy.app.timers.unregister(_setup_jedi)

    utils._suppress_warnings()
    # Support Reload Scripts.
    for name in ("jedi", "parso"):
        if name in sys.modules:
            del sys.modules[name]
            dotted = name + "."
            for name in tuple(sys.modules):
                if name.startswith(dotted):
                    del sys.modules[name]

    # We don't want jedi/parso on sys.path.
    from importlib.util import spec_from_file_location, module_from_spec
    for name in ("parso", "jedi"):
        spec = spec_from_file_location(name, f"{__path__[0]}/{name}/__init__.py")
        spec.loader.exec_module(sys.modules.setdefault(name, module_from_spec(spec)))

    import jedi
    from . import patches

    # Do not let jedi infer anonymous parameters.
    jedi.settings.dynamic_params = False
    jedi.settings.auto_import_modules = set()

    patches.apply()
    settings.loaded = True


def enable():
    utils.register_classes(classes)

    from textension.overrides import default
    default.insert_hooks += on_insert,
    default.delete_hooks += on_delete,

    Suggestions.preferences = prefs.add_settings(TEXTENSION_PG_suggestions)
    update_defaults()

    ui.add_draw_hook(draw_suggestions, draw_index=9)
    ui.add_hit_test(hit_test_suggestions, 'TEXT_EDITOR', 'WINDOW')

    # Override the default auto complete operator.
    TEXT_OT_autocomplete.apply_override()

    # Defer loading jedi and applying patches so the plugin is enabled faster.
    bpy.app.timers.register(_setup_jedi, first_interval=0.3)

    # Allow accessing Suggestions from space data. For introspection.
    bpy.types.SpaceTextEditor.suggestions = property(
        get_instance.__kwdefaults__["cache"].__getitem__)


def disable():
    if prefs.suggestions.show_hover:
        hover.disable()

    from textension.overrides import default
    default.insert_hooks.remove(on_insert)
    default.delete_hooks.remove(on_delete)

    TEXT_OT_autocomplete.remove_override()

    prefs.remove_settings(TEXTENSION_PG_suggestions)
    ui.remove_hit_test(hit_test_suggestions)
    ui.remove_draw_hook(draw_suggestions)
    utils.unregister_classes(classes)
    get_instance.__kwdefaults__["cache"].clear()

    del bpy.types.SpaceTextEditor.suggestions
