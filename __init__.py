import os
import sys
import time

import blf
import bpy
from bpy.types import SpaceTextEditor
from dev_utils import measure

from ... import TEXTENSION_OT_hit_test, gl, types, utils
from ...types import is_spacetext, is_text

system = bpy.context.preferences.system
_context = utils._context

PLUGIN_PATH = os.path.dirname(__file__)
DEFAULT_FONT_SIZE = 14
DEFAULT_LINE_PADDING = 1.45
DEFAULT_TEXT_PADDING = 5
DEFAULT_SCROLLBAR_WIDTH = 20


# TODO: Testing
from dev_utils import enable_breakpoint_hook

enable_breakpoint_hook(True)


class SizeWidget(gl.GLPlainRect):

    def __init__(self, parent: 'Instance', cursor: str):
        super().__init__(0.5, 0.5, 0.5, 0.0)
        self.tag_redraw = parent.region.tag_redraw
        self.cursor = cursor

    def set_alpha(self, value):
        if self.background[3] != value:
            self.background[3] = value
            self.tag_redraw()

    def on_enter(self):
        self.set_alpha(1.0)

    def on_leave(self):
        self.set_alpha(0.0)


class CornerSizeWidget:
    def __init__(self, parent, cursor: str):
        self.cursor = cursor
        self.box = parent.box
        self.resizers = parent.resize_width, parent.resize_height

    def hit_test(self, mrx, mry):
        return mrx >= self.box.x2 - 8 and mry <= self.box.y + 8

    def on_enter(self):
        for resizer in self.resizers:
            resizer.on_enter()

    def on_leave(self):
        for resizer in self.resizers:
            resizer.on_leave()


class EntriesWidget:
    def __init__(self, parent):
        self.parent = parent
    
    def on_leave(self):
        if self.parent.hover_index != -1:
            self.parent.hover_index = -1
            self.parent.region.tag_redraw()

    def on_enter(self):
        pass  # Handled elsewhere.


class Instance:
    box: gl.GLRoundedRect           # Box
    width: int = 300                # Box width
    height: int = 200               # Box height
    visible: bool = False           # Box visibility

    hover: gl.GLRoundedRect         # Entry mouse hover
    gutter: gl.GLRoundedRect        # Scrollbar gutter
    thumb: gl.GLRoundedRect         # Scrollbar thumb
    selection: gl.GLRoundedRect     # Entry selection
    resize_width: SizeWidget        # Width resizer
    resize_height: SizeWidget       # Height resizer
    text_surface: gl.GLTexture      # Entries

    items: tuple = ()               # Completions
    region: bpy.types.Region        # Draw region
    line_height: int = 1            # Line height
    line_heightf: float = 1.0
    active_index: int = 0           # Selected entry
    hover_index: int = -1           # Hovered entry
    clamp_ratio: float = 1.0        # Thumb clamp ratio
    hash: int = 0                   # Completions hash

    cursor_position: tuple[int, int] = (0, 0)

    font_size: int = DEFAULT_FONT_SIZE
    line_padding: float = DEFAULT_LINE_PADDING
    text_padding: int = DEFAULT_TEXT_PADDING
    scrollbar_width: int = DEFAULT_SCROLLBAR_WIDTH

    def __init__(self, st: SpaceTextEditor) -> None :
        self.region = utils.region_from_space_data(st)
        self.hit = None
        self.st = st

        self.box = gl.GLRoundedRect(0.2, 0.2, 0.2, 1.0)
        self.box.set_border_color(0.3, 0.3, 0.3, 1.0)
        self.hover = gl.GLRoundedRect(1.0, 1.0, 1.0, 0.08)
        self.hover.set_border_color(1.0, 1.0, 1.0, 0.08)

        self.gutter = gl.GLRoundedRect(0.18, 0.18, 0.18, 0.0)
        self.gutter.hit_test_func = lambda gutter=self.gutter: bpy.ops.textension.suggestions_scroll('INVOKE_DEFAULT', action=gutter.action)
        self.gutter.on_enter = types.noop
        self.gutter.on_leave = types.noop

        self.thumb = gl.GLRoundedRect(0.27, 0.27, 0.27, 1.0)
        self.thumb.hit_test_func = lambda: bpy.ops.textension.suggestions_scrollbar('INVOKE_DEFAULT')
        self.thumb.on_enter = types.noop
        self.thumb.on_leave = types.noop

        self.selection = gl.GLRoundedRect(0.3, 0.4, 0.8, 0.4)

        self.resize_width = SizeWidget(self, 'MOVE_X')
        self.resize_width.hit_test_func = lambda: bpy.ops.textension.suggestions_resize('INVOKE_DEFAULT', action='HORIZONTAL')

        self.resize_height = SizeWidget(self, 'MOVE_Y')
        self.resize_height.hit_test_func = lambda: bpy.ops.textension.suggestions_resize('INVOKE_DEFAULT', action='VERTICAL')
        
        self.resize_corner = CornerSizeWidget(self, 'SCROLL_XY')
        self.resize_corner.hit_test_func = lambda: bpy.ops.textension.suggestions_resize('INVOKE_DEFAULT', action='CORNER')

        self.entries = EntriesWidget(self)
        self.entries.hit_test_func = lambda instance=self: bpy.ops.textension.suggestions_commit('INVOKE_DEFAULT', index=instance.hover_index)

        self.text_surface = gl.GLTexture(self.width, self.height)
        self.text_surface_cache_key = ()

    def set_top(self, new_top: float) -> None:
        """Assign a new top value."""
        if self.top != new_top:
            self.top = new_top
            self.region.tag_redraw()

    def _validate_indices(self) -> None :
        """If the suggestions list changes, reset top to zero."""
        if self.hash != (_hash := hash(self.items)):
            self.active_index = 0
            self.hash = _hash
            self.top = 0.0

    def test_and_set(self, new_hit):
        """Handle the hit test result's enter/leave events."""
        if (hit := self.hit) != new_hit:
            if hit is not None:
                hit.on_leave()
            self.hit = new_hit
            new_hit.on_enter()
        return new_hit.hit_test_func

    def hit_test_widgets(self, mrx, mry) -> types.Callable | None:
        """Hit test box widgets: resizers, gutter, thumb and entries."""
        # Resizers
        for widget in (self.resize_corner, self.resize_width, self.resize_height):
            if widget.hit_test(mrx, mry):
                _context.window.cursor_set(widget.cursor)
                return self.test_and_set(widget)

        # Gutter
        if self.clamp_ratio != 0 and self.gutter.hit_test(mrx, mry):
            _context.window.cursor_set("DEFAULT")

            # Thumb
            if self.thumb.hit_test(mrx, mry):
                return self.test_and_set(self.thumb)

            self.gutter.action = 'PAGE_DOWN' if mry < self.thumb.y else 'PAGE_UP'
            return self.test_and_set(self.gutter)

        # Entries
        _context.window.cursor_set("DEFAULT")
        hit_index = int(self.top + ((self.box.y2 - mry) / self.line_height))

        if hit_index < len(self.items):
            if hit_index != self.hover_index:
                self.hover_index = hit_index
                self.region.tag_redraw()
            return self.test_and_set(self.entries)

        # No entry was hit, although we are still inside the box.
        # This is possible when the box is taller than all the entries.
        return None

    def poll(self):
        """When this method returns False, the box should not be drawn.

        Predicates include:
        - self.visible must be True
        - The text in the bound space data must not be None
        - The text's cursor position must be synchronized with this instance.
            - Position is updated via the insert/delete operator hooks
        - The instance must have items to show.
        """
        if self.visible and (text := self.st.text) is not None:
            if text.cursor_position == self.cursor_position:
                return self.items
            else:
                self.cursor_position = -1, -1
        return False

    def update_cursor(self):
        self.cursor_position = self.st.text.cursor_position


def instance_from_space(st: SpaceTextEditor, *, cache={}) -> Instance:
    try:
        return cache[st]
    except:
        if not is_spacetext(st):
            raise TypeError(f"Expected a SpaceTextEditor instance, got {st}")
        return cache.setdefault(st, Instance(st))


def clear_instances_cache() -> None:
    """Clear suggestions box instances cache."""
    instance_from_space.__kwdefaults__["cache"].clear()


def test_suggestions_box(data: types.HitTestData) -> types.Callable | None:
    """Hit test hook for TEXTENSION_OT_hit_test."""
    if (ins := instance_from_space(data.space_data)).poll() and \
        ins.box.hit_test(*(pos := data.pos)):
            if (hit := ins.hit_test_widgets(*pos)):
                return hit
            _context.window.cursor_set("DEFAULT")
            return types.noop

    # If a previous hit exists, call its leave handler.
    elif ins.hit is not None:
        ins.hit.on_leave()
        ins.hit = None
    return None


def thumb_vpos_calc(height_px: int, line_px: int, nlines: int, top: float) -> tuple[int, int, float]:
    """Compute the vertical position of the scrollbar thumb.

    height_px: The height of the working range of the scrollbar in pixels.
    line_px:   The line height in pixels.
    nlines:    Total number of lines.
    top:       Current position of the top. Can be float.

    Return:    tuple of y, height and clamped ratio (1.0 if unclamped).

    If the thumb height >= working range, returns 0, 0, 0.
    """
    visible_lines = height_px / line_px

    if visible_lines >= nlines:
        return 0, 0, 0.0

    # Minimum thumb height before it's clamped to keep it clickable.
    min_ratio = min(30, height_px) / height_px
    ratio = visible_lines / nlines

    if ratio > min_ratio:
        y = height_px * (1 - (top + visible_lines) / nlines)
        h = int((height_px * (1 - top / nlines)) - y)
        clamp_ratio = 1.0
    else:
        ymax = height_px - int((height_px * (1 - (min_ratio - ratio))) * top / nlines)
        y = int(ymax - height_px * min_ratio)
        h = ymax - int(ymax - height_px * min_ratio)
        # The height is clamped. The ratio represents the size difference.
        clamp_ratio = min_ratio / ratio

    return max(0, y), h, clamp_ratio


def draw(context: bpy.types.Context):
    """Draw callback for suggestions box."""
    st = context.space_data
    instance = instance_from_space(st)
    # TODO: poll should not return items - should be boolean
    items = instance.poll()
    if not items:
        return

    instance._validate_indices()

    # Box position and size
    wu_scale = system.wu * 0.05
    size = w, h = instance.width, instance.height
    x, y = st.region_location_from_cursor(*st.text.cursor_position)

    # Align the box to below the cursor.
    y -= h - st.offsets[1] - round(4 * wu_scale)

    # At 1.77 scale, dpi is halved and pixel_size is doubled. Don't ask why.
    blf.size(1, instance.font_size, int(system.dpi * system.pixel_size))

    # Line height is computed based on glyph metrics.
    x_height = blf.dimensions(1, "acemnorsuvwxz")[1]
    asc = blf.dimensions(1, "ABC")[1] - x_height
    desc = blf.dimensions(1, "gpqy")[1] - x_height

    top = instance.top
    top_int = int(top)
    line_heightf = instance.line_heightf = (x_height + asc + desc) * instance.line_padding
    line_height = instance.line_height = int(line_heightf)
    offset_px = int((top - top_int) * line_heightf)
    hover_width = w - 2

    # Draw box
    instance.box(x, y, w, h)

    # Draw selection
    sy = y + h - line_height - (line_height * (instance.active_index - top_int)) + offset_px
    sh = line_height
    if sy <= y:  # Selection y is below box
        sh = line_height - (y - sy) - 1
        sy = y + 1
    if sy + sh >= y + h:  # Selection y + height is above box
        sh = line_height - ((sy + sh) - (y + h)) - 1

    instance.selection(x + 1, sy, hover_width, sh)

    # Draw hover
    hy = y + h - line_height - (line_height * (instance.hover_index - top_int)) + offset_px
    hh = line_height
    if hy <= y:
        hh = line_height - (y - hy) - 1
        hy = y + 1
    if hy + hh >= y + h:
        hh = line_height - ((hy + hh) - (y + h)) - 1
    instance.hover(x + 1, hy, hover_width, hh)

    # Draw the text entries
    blf.color(1, 0.4, 0.7, 1.0, 1)
    text_surface = instance.text_surface

    # There's no built-in scissor/clip feature in 'gpu' yet as of 3.2.1, so
    # text is drawn onto a surface and cached using a key.
    if size != text_surface.size:
        text_surface.resize(w, h)

    if (key := (top, instance.hash, line_height, size)) != instance.text_surface_cache_key:
        instance.text_surface_cache_key = key

        max_items = int(h / line_heightf + 2)
        pad = instance.text_padding * wu_scale
        text_x = pad
        text_y = h + desc - line_height + offset_px + (3 * wu_scale)
        with text_surface.bind():
            for compl in items[top_int:top_int + max_items]:
                blf.position(1, text_x, text_y, 0)
                blf.draw(1, compl.name)
                text_y -= line_height

    text_surface(x, y, w, h)

    # Draw thumb
    thumb_y_offset, thumb_h, clamp_ratio = thumb_vpos_calc(h, line_height, len(items), top)
    instance.clamp_ratio = clamp_ratio
    if thumb_h != 0:
        thumb_width = instance.scrollbar_width
        thumb_x = x + w - thumb_width
        thumb_y = y + thumb_y_offset
        instance.gutter(thumb_x, y + 1, thumb_width, h - 2)
        instance.thumb(thumb_x, round(thumb_y), thumb_width, round(thumb_h))

    # Draw sizers even if transparent to update their hit test rectangles.
    instance.resize_width(x + w - 3, y, 3, h)
    instance.resize_height(x, y, w, 3)


class TEXTENSION_OT_suggestions_commit(types.TextOperator):
    bl_options = {'INTERNAL'}
    index: bpy.props.IntProperty(default=-1, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return instance_from_space(context.space_data).poll()

    def execute(self, context):
        instance = instance_from_space(context.space_data)

        is_return = False
        # When self.index is -1 or not given, use the currently active index.
        if self.index == -1:
            assert instance.active_index < len(instance.items)
            self.index = instance.active_index
            is_return = True

        complete_str = instance.items[self.index].complete

        if complete_str:
            text = context.edit_text
            utils.push_undo(text)
            line, column = text.cursor_position
            text.write(complete_str)
            text.cursor = line, column + len(complete_str)
            utils.tag_modified(self)
            instance.visible = False
            context.region.tag_redraw()
            return {'FINISHED'}

        # The string completes nothing. If the commit was via return key, pass through the event.
        else:
            if is_return:
                return {'PASS_THROUGH'}
        instance.visible = False
        context.region.tag_redraw()
        return {'CANCELLED'}

    @classmethod
    def register_keymaps(cls):
        from ...km_utils import kmi_new
        kmi_new(cls, "Text Generic", cls.bl_idname, 'RET', 'PRESS')


class TEXTENSION_OT_suggestions_dismiss(types.TextOperator):
    bl_options = {'INTERNAL'}

    def execute(self, context):
        dismiss()
        return {'CANCELLED'}

    @classmethod
    def register_keymaps(cls):
        from ...km_utils import kmi_new
        kmi_new(cls, "Text", cls.bl_idname, 'ESC', 'PRESS')


class TEXTENSION_OT_suggestions_resize(types.TextOperator):
    bl_options = {'INTERNAL'}
    action: bpy.props.EnumProperty(
        items=(('HORIZONTAL', "Horizontal", "Resize horizontally"),
               ('VERTICAL', "Vertical", "Resize vertically"),
               ('CORNER', "Corner", "Resize from corner")))

    def invoke(self, context, event):
        self.instance = instance_from_space(context.space_data)
        self.width, self.height = self.instance.text_surface.size
        self.min_height = self.instance.line_height
        self.min_width = int(150 * system.wu * 0.05)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return {'CANCELLED'}

        elif event.type == 'MOUSEMOVE':
            instance = self.instance
            x_delta = event.mouse_x - event.mouse_prev_press_x
            y_delta = event.mouse_y - event.mouse_prev_press_y

            *size, = instance.text_surface.size
            if self.action in {'CORNER', 'HORIZONTAL'}:
                size[0] = max(self.min_width, self.width + x_delta)

            if self.action in {'CORNER', 'VERTICAL'}:
                size[1] = max(self.min_height, self.height - y_delta)

                # Resizing vertically past the bottom entry moves the top up.
                if (top := instance.top) > 0:
                    view = (size[1] / instance.line_height)
                    if (bottom := top + view) > len(instance.items):
                        instance.set_top(max(0, top - (bottom - len(instance.items))))

            (instance.width, instance.height) = size
            context.region.tag_redraw()
        return {'RUNNING_MODAL'}



class TEXTENSION_OT_suggestions_navigate(types.TextOperator):
    bl_options = {'INTERNAL'}

    # action: bpy.props.EnumProperty(
    #     items=(('DOWN', "Down", "Navigate down"),
    #            ('UP', "Up", "Navigate up"),
    #            ('PAGE_DOWN', "Page Down", "Navigate Page Down"),
    #            ('PAGE_UP', "Page Up", "Navigate Page Up")))
    action: bpy.props.EnumProperty(
        items=[(v, "", "") for v in ("DOWN", "UP", "PAGE_DOWN", "PAGE_UP")])

    @classmethod
    def poll(cls, context):
        return is_spacetext(st := context.space_data) and \
               instance_from_space(st).poll()

    def execute(self, context):
        instance = instance_from_space(context.space_data)
        # Page is visible lines minus one. Makes it easier to visually track.
        page = int(instance.height / instance.line_height) - 1

        if self.action in {'DOWN', 'UP'}:

            # When only a single item is shown, up/down keys are passed on
            # and the box is closed. We also invalidate the cursor position
            # so that moving back doesn't re-show the box.
            if len(instance.items) == 1:
                instance.cursor_position = -1, -1
                return {'PASS_THROUGH'}

            value = instance.active_index + (1 if self.action == 'DOWN' else -1)
            new_index = value % len(instance.items)
        elif self.action == 'PAGE_DOWN':
            new_index = min(len(instance.items) - 1, instance.active_index + page)
        else:
            new_index = max(0, instance.active_index - page)

        # New index is below bottom
        if new_index >= instance.top + page:
            instance.top = new_index - page

        # New index is above top
        elif new_index < instance.top:
            instance.top = new_index

        instance.active_index = new_index
        context.region.tag_redraw()
        return {'FINISHED'}

    @classmethod
    def register_keymaps(cls):
        from ...km_utils import kmi_new
        kmi_new(cls, "Text", cls.bl_idname, 'UP_ARROW', 'PRESS', repeat=True).action = 'UP'
        kmi_new(cls, "Text", cls.bl_idname, 'DOWN_ARROW', 'PRESS', repeat=True).action = 'DOWN'

        for action in ('PAGE_DOWN', 'PAGE_UP'):
            kmi_new(cls, "Text Generic", cls.bl_idname, action, 'PRESS', repeat=True).action = action


class TEXTENSION_OT_suggestions_scroll(types.TextOperator):
    bl_options = {'INTERNAL'}
    # action: bpy.props.EnumProperty(
    #     items=(('PAGE_DOWN', "Page Down", "Scroll down one page"),
    #            ('PAGE_UP', "Page Up", "Scroll up one page"),
    #            ('LINES', "Lines", "Scroll by amount of lines")))
    action: bpy.props.EnumProperty(
        items=[(v, "", "") for v in ("PAGE_DOWN", "PAGE_UP", "LINES")])
    lines: bpy.props.IntProperty(default=0, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        if types.TextOperator.poll(context):
            region = context.region
            instance = instance_from_space(context.space_data)
            return instance.visible and instance.box.hit_test(
                region.mouse_x, region.mouse_y)

    def invoke(self, context, event):
        self.instance = instance = instance_from_space(context.space_data)
        span = instance.line_height * len(instance.items)
        self.page = instance.height / instance.line_height
        self.max_top = (span / instance.line_height) - self.page

        # Operator was invoked from clicking the gutter.
        if event.type == 'LEFTMOUSE':
            self.action = 'PAGE_DOWN'

            if event.mouse_region_y >= instance.thumb.y:
                self.action = 'PAGE_UP'

            self.scroll(event.mouse_region_y)
            self.delay = time.monotonic() + 0.3
            context.window_manager.modal_handler_add(self)
            self.timer = context.window_manager.event_timer_add(0.01, window=context.window)
            return {'RUNNING_MODAL'}

        # Operator was invoked from scrolling the mouse wheel.
        elif event.type in {'WHEELDOWNMOUSE', 'WHEELUPMOUSE'}:
            lines = 3 if 'DOWN' in event.type else -3
            instance.set_top(max(0, min(self.max_top, instance.top + lines)))
            # Scrolling the entries must update the hover highlights.
            test_suggestions_box(TEXTENSION_OT_hit_test.get_data(context))
        return {'CANCELLED'}

    def modal(self, context, event):
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'RET', 'ESC', 'WINDOW_DEACTIVATE'}:
            context.window_manager.event_timer_remove(self.timer)
            return {'CANCELLED'}

        elif event.type == 'TIMER' and (t := time.monotonic()) >= self.delay:
            if self.scroll(event.mouse_region_y):
                self.delay = t + 0.04

        return {'RUNNING_MODAL'}

    def scroll(self, mry: int):
        top = self.instance.top
        thumb = self.instance.thumb

        if self.action == 'PAGE_DOWN' and mry < thumb.y:
            self.instance.set_top(min(self.max_top, top + self.page))

        elif self.action == 'PAGE_UP' and mry > thumb.y + thumb.height:
            self.instance.set_top(max(0, top - self.page))
        else:
            return False
        return True

    @classmethod
    def register_keymaps(cls):
        from ...km_utils import kmi_new
        prop = kmi_new(cls, "Text Generic", cls.bl_idname, 'WHEELDOWNMOUSE', 'PRESS')
        prop.lines = 3
        prop.action = 'LINES'
        prop = kmi_new(cls, "Text Generic", cls.bl_idname, 'WHEELUPMOUSE', 'PRESS')
        prop.lines = -3
        prop.action = 'LINES'


class TEXTENSION_OT_suggestions_scrollbar(types.TextOperator):
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        instance = self.instance = instance_from_space(context.space_data)
        height = instance.height
        line_height = instance.line_height
        span = line_height * len(instance.items)
        self.max_top = (span / line_height) - (height / line_height)  # maximum scrollable (span minus 1 page)
        self.px_ratio = span / height * 1.015  # Nasty compensation hack
        self.top_org = instance.top
        self.max_px = self.max_top * line_height
        self.line_height_coeff = 1 / line_height
        min_height = min(30, height)
        self.clamp_diff_px = (min_height - (min_height / instance.clamp_ratio)) * self.px_ratio
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return {'CANCELLED'}

        elif event.type == 'MOUSEMOVE':
            if abs(event.mouse_prev_press_x - event.mouse_x) > 140:
                top = self.top_org
            else:
                mouse_y_delta = event.mouse_prev_press_y - event.mouse_y
                scroll_px_delta = mouse_y_delta * self.px_ratio

                # If the scroll thumb is clamped, multiply the difference against
                # scroll pixels and apply it linearly towards maximum pixel span.
                scroll_px_delta += scroll_px_delta * self.clamp_diff_px / self.max_px
                line_delta = scroll_px_delta * self.line_height_coeff
                top = max(0, min(self.max_top, self.top_org + line_delta))
            self.instance.set_top(top)

        return {'RUNNING_MODAL'}


def dismiss():
    context = bpy.context
    instance = instance_from_space(context.space_data)
    if instance.visible:
        instance.visible = False
        TEXTENSION_OT_hit_test.poll(context)
        context.region.tag_redraw()
# from jedi.api import Interpreter2
# from jedi import Interpreter

# def complete(st):
#     text = st.text
#     instance = instance_from_space(st)
#     line, col = text.cursor_position
#     instance.update_cursor()
#     string = text.as_string()
#     # print()
#     # with measure:
#     #     bpy.interp = interp = Interpreter(string, [])
#     #     instance.items = tuple(interp.complete(line + 1, col))

#     with measure:
#         bpy.interp = interp = Interpreter2(string)
#         instance.items = tuple(interp.complete_unsafe(line + 1, col))

#     TEXTENSION_OT_hit_test.poll(_context)
#     instance.visible = True
#     instance.region.tag_redraw()


separators = {*" !\"#$%&\'()*+,-/:;<=>?@[\\]^`{|}~"}  # Excludes "."
"""
Token separators that determine when completions should show after
a character has been inserted. If the character matches the token,
completions will not show.
"""


def on_insert() -> None:
    """Called after TEXTENSION_OT_insert.

    Completion shouldn't run when the inserted character is a token separator
    excluding period "."
    """
    st = _context.space_data
    line, col = st.text.cursor_position
    if st.text.lines[line].body[col - 1] in separators:
        dismiss()
        return

    # Sometimes jedi can be slow. We don't want te insert operator to freeze
    # until it completes, so completions run in the next event loop iteration.
    else:
        bpy.ops.textension.suggestions_complete('INVOKE_DEFAULT')


def deferred_complete(st: bpy.types.SpaceTextEditor):
    override = bpy.context.copy()
    override["space_data"] = st
    def wrapper():
        bpy.ops.textension.suggestions_complete(override)
    bpy.app.timers.register(wrapper, first_interval=1e-4)


def on_delete() -> None:
    """Called after backspace operator runs. Completions run and show only if
    there's text leading up to the cursor and it doesn't have a trailing comma.
    """
    st = _context.space_data
    text = _context.edit_text

    line, col = st.text.cursor_position
    lead_text = text.lines[line].body[:col]

    if not lead_text.strip():
        dismiss()
    elif lead_text[col - 1: col] in separators:
        dismiss()
    elif lead_text.endswith("."):
        dismiss()
    elif text.lines[line].body[col - 1:col] in {"(", "[", "{", "\"", "\'"}:
        dismiss()

    elif lead_text.lstrip():
        instance = instance_from_space(st)
        instance.update_cursor()
        # If the box already is visible, run completions again.
        if instance.poll():
            bpy.ops.textension.suggestions_complete('INVOKE_DEFAULT')


class TEXTENSION_OT_suggestions_complete(types.TextOperator):
    @classmethod
    def poll(cls, context):
        return is_spacetext(context.space_data) and \
               is_text(getattr(context, "edit_text", None))

    def execute(self, context):
        st = context.space_data
        if not is_spacetext(st):
            raise Exception(f"Expected a SpaceTextEditor instance, got {st}")
        text = st.text
        instance = instance_from_space(st)
        line, col = text.cursor_position
        instance.update_cursor()
        string = text.as_string()

        # TODO: If optimized, use that version.
        # from jedi.api import Interpreter
        from .optimizations import Interpreter2 as Interpreter, _clear_caches
        
        # ret = sorted(interp.complete(line + 1, col, fuzzy=True),
        #     key=lambda x:x.complete is not None, reverse=False)
        # with measure:
        # _clear_caches()
        interp = Interpreter(string, [])
        ret = interp.complete_unsafe(line + 1, col)

        import gc
        _clear_caches()
        gc.collect()
        interp._inference_state.memoize_cache.clear()
        try:
            last = bpy.last
        except:
            last = bpy.last = len(gc.get_objects())
        else:
            bpy.last = curr = len(gc.get_objects())
            print(curr, last, curr - last, "new")

            # ret = interp.complete(line + 1, col)
        # ret = interp.complete_unsafe(line + 1, col)
        instance.items = tuple(ret)

        TEXTENSION_OT_hit_test.poll(_context)
        instance.visible = True
        instance.region.tag_redraw()
        return {'CANCELLED'}

    def invoke(self, context, event):
        st = context.space_data
        instance_from_space(st).update_cursor()
        deferred_complete(st)
        return {'FINISHED'}

    @classmethod
    def register_keymaps(cls):
        from ...km_utils import kmi_new
        kmi_new(cls, "Text Generic", cls.bl_idname, 'SPACE', 'PRESS', ctrl=1)


class TEXTENSION_OT_suggestions_download_jedi(bpy.types.Operator):
    """Download Jedi and other dependencies using pip"""

    bl_idname = "textension.suggestions_download_jedi"
    bl_label = "Download Jedi"

    @classmethod
    def poll(cls, context):
        return context.area.type == 'PREFERENCES'

    def execute(self, context, *, g={"active": False}):

        if g["active"]:
            print("download alread active, cancelling")
            return {'CANCELLED'}

        # Uses threading to ensure the ui doesn't hang while we attempt
        # to download jedi/parso via pip.
        import threading
        import _socket
        import queue
        import signal

        # Hacky, but we want a connection through blender, not python.
        from pip._internal.cli.main import main as pipmain

        print("\n" * 10)
        g["active"] = True

        S_CONNECTING = -1
        S_CONNECTED = 1
        S_ERROR = 3
        S_DOWNLOADING = 4
        S_COMPLETE = 5
        S_NOCONNECTION = 6

        work = queue.Queue()
        conn = _socket.socket()
        event = threading.Event()
        event.clear()

        def set_status(status):
            if status == S_NOCONNECTION:
                print("Connection failed")
                g["active"] = False
                return
            elif status == S_CONNECTED:
                print("Connected")
                return
            elif status == S_DOWNLOADING:
                print("Downloading...")
                return
            elif status == S_COMPLETE:
                print("Download OK")

                poll_plugin.cache.clear()
                utils.redraw_editors(area='PREFERENCES')
                g["active"] = False
                work.put(lambda: False)
            elif status == S_CONNECTING:
                print("Connecting...")

            elif status == S_ERROR:
                print("Error...")
                g["active"] = False
            else:
                print("Unhandled status:", status)

        from bpy.app.timers import register as register_timer

        def worker_main():
            register_timer(lambda s=S_CONNECTING: set_status(s))
            retries = 5
            connected = False
            address = ("pypi.org", 443)
            while not connected:
                try:
                    conn.connect(address)
                except OSError as e:
                    if retries <= 0:
                        # print(f"Jedi download failed: \n    {e}")
                        register_timer(lambda s=S_NOCONNECTION: set_status(s))
                        break
                    else:
                        retries -= 1
                        time.sleep(1.0)
                else:
                    conn.close()
                    register_timer(lambda s=S_CONNECTED: set_status(s))
                    connected = True

                    register_timer(lambda s=S_DOWNLOADING: set_status(s))
                    ret = pipmain(["install", "jedi", "-t", PLUGIN_PATH, "--upgrade"])
                    if ret == 0:
                        register_timer(lambda s=S_COMPLETE: set_status(s))

            print("worker thread: exiting...")
            return None

        # bpy.app.timers.register(timer_main)
        dl_thread = threading.Thread(target=worker_main)
        print("starting worker thread..")
        dl_thread.start()
        return {'FINISHED'}

def iter_instances():
    for space in utils.iter_spaces(space_type='TEXT_EDITOR'):
        yield instance_from_space(space)


def redraw_open_instances():
    for instance in iter_instances():
        if instance.visible:
            instance.region.tag_redraw()



class TEXTENSION_PG_suggestions_settings(bpy.types.PropertyGroup):

    def on_update_font_size(self, _):
        Instance.font_size = self.font_size

    def on_update_line_padding(self, _):
        Instance.line_padding = self.line_padding

    def on_update_text_padding(self, _):
        # Invalidate text surface so it can be redrawn
        for instance in iter_instances():
            instance.text_surface_cache_key = None
        Instance.text_padding = self.text_padding

    def on_update_scrollbar_width(self, _):
        Instance.scrollbar_width = self.scrollbar_width

    font_size: bpy.props.IntProperty(
        name="Font Size",
        description="Font size for suggestions entries",
        update=on_update_font_size,
        default=DEFAULT_FONT_SIZE,
        min=1,
        max=144)

    line_padding: bpy.props.FloatProperty(
        name="Line Padding",
        description="Line Padding",
        update=on_update_line_padding,
        default=DEFAULT_LINE_PADDING,
        min=0.0,
        max=4.0)

    text_padding: bpy.props.IntProperty(
        name="Text Padding",
        description="Text Padding",
        update=on_update_text_padding,
        default=DEFAULT_TEXT_PADDING,
        min=0,
        max=1000)

    scrollbar_width: bpy.props.IntProperty(
        name="Scrollbar Width",
        description="Scrollbar Width",
        update=on_update_scrollbar_width,
        default=DEFAULT_SCROLLBAR_WIDTH,
        min=8,
        max=100)

classes = (
    TEXTENSION_PG_suggestions_settings,
    TEXTENSION_OT_suggestions_complete,
    TEXTENSION_OT_suggestions_scrollbar,
    TEXTENSION_OT_suggestions_scroll,
    TEXTENSION_OT_suggestions_navigate,
    TEXTENSION_OT_suggestions_resize,
    TEXTENSION_OT_suggestions_dismiss,
    TEXTENSION_OT_suggestions_commit,
)


def enable():
    bpy.utils.register_class(TEXTENSION_OT_suggestions_download_jedi)

    # Unless Jedi already exists, it's placed into the directory 'download'.
    # In this case we add it to sys.path to make it globally importable.
    if PLUGIN_PATH not in sys.path:  # TODO: Should be 'download', not root directory.
        sys.path.append(PLUGIN_PATH)

    if not poll_plugin():
        return

    import jedi
    for cls in classes:
        bpy.utils.register_class(cls)

    from ...utils import TextensionPreferences
    TextensionPreferences.suggestions = bpy.props.PointerProperty(type=TEXTENSION_PG_suggestions_settings)

    from ... import TEXTENSION_OT_insert, TEXTENSION_OT_delete
    TEXTENSION_OT_insert.insert_hooks.append(on_insert)
    TEXTENSION_OT_delete.delete_hooks.append(on_delete)

    # Include module names which jedi can't infer on its own
    jedi.settings.auto_import_modules[:] = set(
        jedi.settings.auto_import_modules + ["bpy", "_bpy", "numpy", "sys"])

    utils.add_draw_hook(draw, SpaceTextEditor, (_context,))
    utils.add_hittest(test_suggestions_box)

    from .optimizations import setup2
    setup2()


_poll_cache = [False, 0]


def poll_plugin():
    """Return whether parso/jedi are importable, without importing them.
    Optimized to be usable in a draw function.
    """
    try:
        poll, timeout = _poll_cache
    except:
        poll, timeout = _poll_cache[:] = (False, 0)

    if not poll and timeout < (curr_time := time.monotonic()):
        from importlib.util import find_spec
        poll = bool(find_spec("parso") and find_spec("jedi"))
        _poll_cache[:] = (poll, curr_time + 2.0)
    return poll


def draw_settings(prefs, context, layout):
    if not poll_plugin():
        layout.separator()
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="This plugin needs Jedi (5MB)")

        layout.separator()

        row = layout.row()
        row.alignment = 'CENTER'
        row.operator("textension.suggestions_download_jedi")
        layout.separator()

def disable():
    bpy.utils.unregister_class(TEXTENSION_OT_suggestions_download_jedi)

    from ...utils import TextensionPreferences
    del TextensionPreferences.suggestions

    from ... import TEXTENSION_OT_insert, TEXTENSION_OT_delete
    TEXTENSION_OT_insert.insert_hooks.remove(on_insert)
    TEXTENSION_OT_delete.delete_hooks.remove(on_delete)

    utils.remove_hittest(test_suggestions_box)
    utils.remove_draw_hook(draw)
    clear_instances_cache()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

# import lark
