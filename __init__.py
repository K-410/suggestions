import os
import sys
import time

import blf
import bpy
from bpy.types import SpaceTextEditor

from ... import TEXTENSION_OT_hit_test, gl, types, utils, _iter_expand_tokens
from ...types import is_spacetext, is_text
from ...km_utils import keymap
from ... import ui
# TODO: Testing

# from dev_utils import enable_breakpoint_hook, per, measure, total, accumulate
# import threading, debugpy
# debugpy.configure(subProcess=False)
# threading.Thread(target=debugpy.listen, args=(5678,), daemon=True).start()
# print("Debug server at 127.0.0.1:5678")

_context = utils._context
system = _context.preferences.system
PLUGIN_PATH = os.path.dirname(__file__)

# Separators determine when completions should show after character insertion.
# If the character matches the separator, completions will not run.
separators = {*" !\"#$%&\'()*+,-/:;<=>?@[\\]^`{|}~"}  # Excludes "."


def test_and_update(obj, attr, value) -> bool:
    if getattr(obj, attr) != value:
        return not setattr(obj, attr, value)
    return False


class WidgetBase:
    prefs: bpy.types.PropertyGroup

    def enter(self):
        return None

    def leave(self):
        return None

    def activate(self):
        return None


class Widget(WidgetBase):
    cursor: str

    def __init__(self):
        self.rect = gl.GLRect()
        self.update = self.rect.update

    def hit_test(self, x: float, y: float) -> bool:
        """Hit test this Widget. Assumes x/y is in region space."""
        rect = self.rect
        return 0.0 <= x - rect[0] <= rect[2] and 0.0 <= y - rect[1] <= rect[3]



def set_thumb_highlight(widget: Widget, on: bool):
    bg = widget.prefs.thumb_background_color
    bd = widget.prefs.thumb_border_color

    bg, bd = ((min(1.0, v + (0.05 * on)) for v in c) for c in (bg, bd))
    widget.update(background_color=bg, border_color=bd)
    _context.area.tag_redraw()


class Scrollbar(Widget):
    action: str
    clamp_ratio   = 0.0
    thumb_coords  = (0, 0)
    thumb_compute_args  = (0, 0, 0, 0.0)

    thumb_highlight  = 0.34, 0.34, 0.34, 1.0

    def __init__(self, listbox: "ListBox"):
        super().__init__()
        self.listbox = listbox
        self.thumb   = thumb = Widget()

        self.update(background_color=(0.0, 0.0, 0.0, 0.0),
                    border_color    =(0.0, 0.0, 0.0, 0.0),
                    radius          =self.prefs.radius)
        thumb.update(background_color=self.prefs.thumb_background_color,
                     border_color    =self.prefs.thumb_border_color,
                     radius          =self.prefs.radius)

        thumb.enter = lambda: set_thumb_highlight(thumb, True)
        thumb.leave = lambda: set_thumb_highlight(thumb, False)
        thumb.activate = lambda: bpy.ops.textension.suggestions_scrollbar('INVOKE_DEFAULT')

    def activate(self):
        bpy.ops.textension.suggestions_scroll('INVOKE_DEFAULT', action=self.action)

    def hit_test(self, mrx, mry):
        if self.clamp_ratio != 0.0 and super().hit_test(mrx, mry):
            self.action = 'PAGE_DOWN' if mry < self.thumb.rect.y else 'PAGE_UP'
            if self.thumb.hit_test(mrx, mry):
                return self.thumb
            return self

    def draw(self):
        ty, th = self.compute_thumb()
        if th != 0:
            lbox = self.listbox

            sx, sy = lbox.instance.rect.inner_position
            sw = lbox.prefs.scrollbar_width * (system.wu * 0.05)
            sx += lbox.width - sw

            self.rect.draw(sx, sy, sw, lbox.height)
            self.thumb.rect.draw(sx, sy + ty, sw, th)

    def compute_thumb(self):
        """Compute the vertical position of the scrollbar thumb."""
        listbox = self.listbox
        thumb_compute_args = (listbox.height, len(listbox.items), listbox.top)
        if test_and_update(self, "thumb_compute_args", thumb_compute_args):
            height, nlines, top = thumb_compute_args
            visible_lines = listbox.visible_lines
            if visible_lines >= nlines:
                self.clamp_ratio = 0.0
                self.thumb_coords = (0, 0)
            else:
                # Minimum thumb height before it's clamped to keep it clickable.
                min_ratio = min(30, height) / height
                ratio = visible_lines / nlines

                if ratio > min_ratio:
                    y = int(height * (1 - listbox.bottom / nlines))
                    h = int((height * (1 - top / nlines)) - y)
                    self.clamp_ratio = 1.0
                else:
                    ymax = height - int((height * (1 - (min_ratio - ratio))) * top / nlines)
                    y = int(ymax - height * min_ratio)
                    h = ymax - int(ymax - height * min_ratio)
                    # The height is clamped. The ratio represents the size difference.
                    self.clamp_ratio = min_ratio / ratio
                if y < 0:
                    y = 0
                self.thumb_coords = (y, h)
        return self.thumb_coords


class Resizer(Widget):
    def __init__(self, instance: 'Instance', cursor: str, action: str):
        super().__init__()
        self.update(background_color=(0.5, 0.5, 0.5, 0.0),
                    border_color    =(0.5, 0.5, 0.5, 0.0))

        self.instance = instance
        self.action = action
        self.cursor = cursor
        self.sizers = [self]

    def set_alpha(self, value):
        for resizer in self.sizers:
            if test_and_update(resizer.rect.background_color, "w", value):
                self.instance.region.tag_redraw()

    def enter(self):
        if self.prefs.show_resize_highlight:
            self.set_alpha(1.0)

    def leave(self):
        self.set_alpha(0.0)

    def activate(self):
        bpy.ops.textension.suggestions_resize('INVOKE_DEFAULT', action=self.action)


class BoxResizer(Resizer):
    def __init__(self, instance: "Instance"):
        super().__init__(instance, 'SCROLL_XY', 'CORNER')
        self.horz = Resizer(instance, 'MOVE_X', 'HORIZONTAL')
        self.vert = Resizer(instance, 'MOVE_Y', 'VERTICAL')
        self.sizers[:] = (self.horz, self.vert)

    def hit_test(self, mrx, mry):
        if mrx >= self.instance.rect.x2 - 8 and mry <= self.instance.rect[1] + 8:
            return self
        for s in self.sizers:
            if s.hit_test(mrx, mry):
                return s
        return None

    def draw(self, x, y, w, h):
        self.horz.rect.draw(x + w - 4, y, 5, h)
        self.vert.rect.draw(x, y - 1, w, 5)


class EntryRect(Widget):
    index: int = 0
    show = True

    def enter(self):
        self.show = True
        _context.region.tag_redraw()

    def leave(self):
        self.show = False
        _context.region.tag_redraw()

    def draw(self, x, y, w, h):
        if self.show:
            self.rect.draw(x, y, w, h)

class ListBox(WidgetBase):
    cache_key: tuple = ()
    scrollbar: Scrollbar
    line_height: int     = 1
    metrics_key: tuple[int, int] = (0, 0)

    def __init__(self, instance: "Instance"):
        super().__init__()
        self.instance = instance
        self.hash   = 0
        self.items  = ()        # Completions
        self.top    = 0.0       # Scroll position
        self.x      = 0
        self.y      = 0

        # Compensate for the top/bottom 1px border of the box.
        self.surface_size = self.width, self.height = round(instance.rect.inner_width), round(instance.rect.inner_height)
        assert self.width > 0
        assert self.height > 0
        self.surface = gl.GLTexture(self.width, self.height)

        self.scrollbar = Scrollbar(listbox=self)

        self.active = EntryRect()
        self.active.update(background_color=self.prefs.active_background_color,
                           border_color    =self.prefs.active_border_color,
                           border_width    =self.prefs.active_border_width,
                           radius          =self.prefs.radius)

        self.hover = EntryRect()
        self.hover.update(background_color=self.prefs.hover_background_color,
                          border_color    =self.prefs.hover_border_color,
                          border_width    =self.prefs.hover_border_width,
                          radius          =self.prefs.radius)
        self.hover.activate = lambda: bpy.ops.textension.suggestions_commit('INVOKE_DEFAULT')
    def leave(self):
        if test_and_update(self.hover, "index", -1):
            self.instance.region.tag_redraw()

    def activate(self):
        # Left mouse pressed
        self.active.index = self.hover.index
        bpy.ops.textension.suggestions_commit('INVOKE_DEFAULT')

    def draw(self):
        instance = self.instance
        blf.color(1, *self.prefs.foreground_color)

        # There's no scissor/clip as of 3.2, so we draw to an off-screen
        # surface instead. The surface is cached for performance reasons.
        size = (w, h) = (self.width, self.height) = round(instance.rect.inner_width), round(instance.rect.inner_height)
        (x, y) = self.x, self.y = round(instance.rect.inner_x), round(instance.rect.inner_y)
        if test_and_update(self, "surface_size", size):
            self.surface.resize(w, h)

        # If the items changed, reset scrollbar, active and hover
        if test_and_update(self, "hash", hash(self.items)):
            self.active.index = 0
            self.hover.index = -1  # Hover is visible only on cursor hover.
            self.top = 0.0

        # At 1.77 scale, dpi is halved and pixel_size is doubled. Go figure.
        blf.size(1, self.font_size, int(system.dpi * system.pixel_size))
        _, adh = blf.dimensions(1, "Ag")
        line_height_f = adh * self.prefs.line_padding
        self.line_height = int(line_height_f)

        lh = self.line_height
        top = self.top
        # The offset in pixels into the current top index
        offset = top % 1 * line_height_f
        top_int = int(top)
        if test_and_update(self, "cache_key", (top, self.hash, lh, size)):
            _, xh = blf.dimensions(1, "x")
            max_items = int(h / self.line_height) + 2
            text_x_origin = self.prefs.text_padding * system.wu * 0.05
            diff = line_height_f - adh
            desc = (adh - xh) * 0.5  # Averaged ascender/descender
            text_y = (h - 1) + desc - lh + offset + (diff * 0.5)

            with self.surface.bind():
                bpy.ret = self.items
                for item in self.items[top_int:top_int + max_items]:
                    text_x = text_x_origin

                    blf.position(1, text_x, text_y, 0)

                    name = item.name
                    length = item.get_completion_prefix_length()
                    if length != 0:
                        blf.color(1, *self.prefs.match_foreground_color)

                        prefix = name[:length]
                        blf.draw(1, prefix)
                        # Draw again shifted 1px to make it bold.
                        # blf.position(1, text_x + 1, text_y, 0)
                        # blf.draw(1, prefix)

                        text_x += blf.dimensions(1, prefix)[0]
                        blf.position(1, text_x, text_y, 0)
                        name = name[length:]
                    blf.color(1, *self.prefs.foreground_color)
                    blf.draw(1, name)
                    text_y -= lh

        for widget in (self.active, self.hover):
            height = lh
            ypos = y + h - height - (height * (widget.index - top_int)) + offset
            if ypos <= y:  # active y is below box
                height -= (y - ypos)
                ypos = y
            if ypos + height >= y + h:  # active y + height is above box
                height -= (ypos + height - (y + h))
            widget.draw(x, ypos, w, height)

        self.surface(x, y, w, h)
        self.scrollbar.draw()

    def hit_test(self, mrx, mry):
        if hit := self.scrollbar.hit_test(mrx, mry):
            return hit

        # Mouse is inside horizontally
        if 0 <= (mrx - self.surface.x) <= self.width:
            top = self.top
            # The view-space hit index
            hit = self.top + (self.y + self.height - mry) / self.line_height
            # Mouse is inside vertically
            if self.top <= hit < min(len(self.items), self.bottom):
                if test_and_update(self.hover, "index", int(hit)):
                    self.instance.region.tag_redraw()
                return self.hover
        return None

    def set_top(self, top: float) -> None:
        """Assign a new top value."""
        if test_and_update(self, "top", max(0, min(top, self.max_top))):
            self.instance.region.tag_redraw()

    @property
    def bottom(self):
        return self.top + self.visible_lines

    @property
    def max_top(self):
        return max(0, len(self.items) - self.visible_lines)

    @property
    def visible_lines(self) -> float:
        return self.height / self.line_height

    @property
    def font_size(self):
        if self.prefs.auto_font_size:
            return self.instance.st.font_size
        return self.prefs.font_size


class Instance(Widget):
    hit = None
    visible = False
    cursor_position = (0, 0)
    st: bpy.types.SpaceTextEditor

    def __init__(self, st: SpaceTextEditor):
        assert is_spacetext(st), "Not SpaceTextEditor, got %s" % st
        super().__init__()

        self.update(
            rect=(0, 0, 300, 200),
            background_color=self.prefs.background_color,
            border_color=self.prefs.border_color,
            border_width=self.prefs.border_width,
            radius=self.prefs.radius)

        self.st = st
        self.entries = ListBox(self)
        self.resizer = BoxResizer(self)

    # Regions can recycle space datas, so always get dynamically.
    region = property(lambda self: utils.region_from_space_data(self.st))

    def poll(self) -> bool:
        if self.visible:
            if text := _context.space_data.text:
                if text.cursor_position == self.cursor_position:
                    return bool(self.entries.items)
                else:
                    self.cursor_position = -1, -1
        return False

    def sync_cursor(self):
        """Synchronize the cursor value with the text (required by poll)."""
        self.cursor_position = _context.space_data.text.cursor_position

    def hit_test(self, mrx, mry):
        # Test if we're inside the box
        if super().hit_test(mrx, mry):
            # Test if we're inside the resize handles
            if hit := self.resizer.hit_test(mrx, mry):
                _context.window.cursor_set(hit.cursor)
            else:
                # Test if we're hitting any entry
                _context.window.cursor_set("DEFAULT")
                hit = self.entries.hit_test(mrx, mry)

            self.set_new_hit(hit)
            if hit:
                return hit.activate

            _context.window.cursor_set("DEFAULT")
            return types.noop
        return None

    def set_new_hit(self, hit=None):
        if self.hit is not hit:
            if self.hit is not None:
                self.hit.leave()
            self.hit = hit
            if hit is not None:
                hit.enter()

    def draw(self):
        st = _context.space_data
        x, y = st.region_location_from_cursor(*st.text.cursor_position)
        # Align the box below the cursor.
        caret_offset = round(4 * system.wu * 0.05)
        y -= self.rect[3] - st.offsets[1] - caret_offset

        self.rect.draw(x, y, self.rect[2], self.rect[3])
        self.entries.draw()
        self.resizer.draw(x, y, self.rect[2], self.rect[3])


def get_instance(*, cache={}) -> Instance:
    try:
        return cache[_context.space_data]
    except KeyError:
        st = _context.space_data
        return cache.setdefault(st, Instance(st))


def test_suggestions_box(x, y) -> types.Callable | None:
    """Hit test hook for TEXTENSION_OT_hit_test."""
    instance = get_instance()
    if instance.poll():
        ret = instance.hit_test(x, y)
        # If a previous hit exists, call its leave handler.
        if ret in {types.noop, None}:
            instance.set_new_hit(None)
        return ret
    return None


def draw():
    """Draw callback for suggestions box."""
    instance = get_instance()
    if instance.poll():
        instance.draw()


@classmethod
def instance_poll(cls, context):
    if is_spacetext(context.space_data):
        return get_instance().poll()
    return False


class TEXTENSION_OT_suggestions_commit(types.TextOperator):
    keymap("Text Generic", 'RET', 'PRESS')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    poll = instance_poll

    def execute(self, context):
        instance = get_instance()
        entries = instance.entries
        is_return = True

        index = entries.active.index

        if index == -1:  # Use active index in entries
            index = entries.hover.index
            assert 0 < index < len(entries.items)
            is_return = False

        text = context.edit_text
        line, col = text.cursor_position

        # The selected completion.
        item = entries.items[index]

        # The name which jedi is being asked to complete.
        word_start = col - item.get_completion_prefix_length()
        projected = text.lines[line].body[word_start:col] + item.complete

        # The query + completion, including parameter/function suffixes.
        completion = item.name_with_symbols

        # Complete only if it either adds to, or modifies the query.
        if item.complete or projected != completion:
            text.curc = word_start
            text.write(completion)
            text.cursor = line, col + len(item.complete)
            instance.visible = False
            context.region.tag_redraw()
            return {'FINISHED'}

        # The string completes nothing. If the commit was via return key, pass through the event.
        else:
            if is_return:
                instance.visible = False
                return {'PASS_THROUGH'}
        instance.visible = False
        context.region.tag_redraw()
        return {'CANCELLED'}


class TEXTENSION_OT_suggestions_dismiss(types.TextOperator):
    keymap("Text", 'ESC', 'PRESS')
    bl_options = {'INTERNAL'}

    def execute(self, context):
        dismiss()
        return {'CANCELLED'}


class TEXTENSION_OT_suggestions_resize(types.TextOperator):
    bl_options = {'INTERNAL'}
    action: bpy.props.EnumProperty(
        items=(('HORIZONTAL', "Horizontal", "Resize horizontally"),
               ('VERTICAL', "Vertical", "Resize vertically"),
               ('CORNER', "Corner", "Resize from corner")))

    def invoke(self, context, event):
        self.instance = get_instance()
        self.start_width, self.start_height = self.instance.rect.size
        self.min_height = self.instance.entries.line_height
        self.min_width = int(150 * system.wu * 0.05)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            # Clear resize hover if the edges were clamped
            if not self.instance.entries.scrollbar.hit_test(
                event.mouse_region_x, event.mouse_region_y):
                self.instance.set_new_hit(None)
            return {'CANCELLED'}

        elif event.type == 'MOUSEMOVE':
            instance = self.instance
            x_delta = event.mouse_x - event.mouse_prev_press_x
            y_delta = event.mouse_y - event.mouse_prev_press_y

            x, y = instance.rect.position
            w, h = instance.rect.size

            if self.action in {'CORNER', 'HORIZONTAL'}:
                w = max(self.min_width, self.start_width + x_delta)
                # Clamp width
                w -= max(0, (x + w) - context.region.width)
                

            if self.action in {'CORNER', 'VERTICAL'}:
                rh = context.region.height
                # Clamp height
                max_height = min(rh - (rh - (y + h)), self.start_height - y_delta)
                h = max(self.min_height, max_height)

                # Vertically resizing past the bottom shifts the top.
                entries = instance.entries
                if entries.top > 0:
                    # TODO: Not sure why -2 is needed.
                    visible_lines = (h - 2) / entries.line_height
                    shift = entries.top + visible_lines - len(entries.items)
                    if shift > 0.0:
                        entries.set_top(entries.top - shift)
            if test_and_update(instance.rect, "size", (w, h)):
                context.region.tag_redraw()
        return {'RUNNING_MODAL'}



class TEXTENSION_OT_suggestions_navigate(types.TextOperator):
    keymap("Text Generic", 'UP_ARROW', 'PRESS', repeat=True, action='UP')
    keymap("Text Generic", 'DOWN_ARROW', 'PRESS', repeat=True, action='DOWN')
    keymap("Text Generic", 'PAGE_UP', 'PRESS', repeat=True, action='PAGE_UP')
    keymap("Text Generic", 'PAGE_DOWN', 'PRESS', repeat=True, action='PAGE_DOWN')

    bl_options = {'INTERNAL'}

    action: bpy.props.EnumProperty(
        items=[(v, v, v) for v in ("DOWN", "UP", "PAGE_DOWN", "PAGE_UP")])

    poll = instance_poll

    def execute(self, context):
        instance = get_instance()
        entries = instance.entries
        # Page is visible lines minus one. Makes it easier to visually track.
        page = int(entries.visible_lines) - 1

        if self.action in {'DOWN', 'UP'}:
            # When only a single item is shown, up/down keys are passed on
            # and the box is closed. We also invalidate the cursor position
            # so that moving back doesn't re-show the box.
            if len(entries.items) == 1:
                instance.cursor_position = -1, -1
                return {'PASS_THROUGH'}

            value = entries.active.index + (1 if self.action == 'DOWN' else -1)
            new_index = value % len(entries.items)
        elif self.action == 'PAGE_DOWN':
            new_index = min(len(entries.items) - 1, entries.active.index + page)
        else:
            new_index = max(0, entries.active.index - page)

        if new_index > entries.top + page:  # New index is below bottom
            # Align the bottom line to the edge
            rem = (entries.height / entries.line_height) % 1.0
            # entries.top = min(entries.top + entries.bottom, new_index - page)
            entries.top = (new_index - page) - rem
        elif new_index < entries.top:        # New index is above top
            entries.top = new_index

        entries.active.index = new_index
        context.region.tag_redraw()
        return {'FINISHED'}


class TEXTENSION_OT_suggestions_scroll(types.TextOperator):
    keymap("Text Generic", 'WHEELDOWNMOUSE', 'PRESS', lines=3, action='LINES')
    keymap("Text Generic", 'WHEELUPMOUSE', 'PRESS', lines=-3, action='LINES')

    bl_options = {'INTERNAL'}
    action: bpy.props.EnumProperty(
        items=[(v, "", "") for v in ("PAGE_DOWN", "PAGE_UP", "LINES")])
    lines: bpy.props.IntProperty(default=0, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        if types.TextOperator.poll(context):
            region = context.region
            instance = get_instance()
            return instance.poll() and instance.hit_test(region.mouse_x, region.mouse_y)

    def invoke(self, context, event):
        self.entries = entries = get_instance().entries
        self.thumb_rect = entries.scrollbar.thumb.rect

        # Operator was invoked from clicking the gutter.
        if event.type == 'LEFTMOUSE':
            self.action = 'PAGE_DOWN'

            if event.mouse_region_y >= self.thumb_rect.y:
                self.action = 'PAGE_UP'

            self.scroll(event.mouse_region_y)
            self.delay = time.monotonic() + 0.3
            context.window_manager.modal_handler_add(self)
            self.timer = context.window_manager.event_timer_add(0.01, window=context.window)
            return {'RUNNING_MODAL'}

        # Operator was invoked from scrolling the mouse wheel.
        elif event.type in {'WHEELDOWNMOUSE', 'WHEELUPMOUSE'}:
            lines = 3 if 'DOWN' in event.type else -3
            entries.set_top(entries.top + lines)
            # Scrolling the entries must update the hover highlights.
            test_suggestions_box(*ui.get_mouse_region())
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
        entries = self.entries
        if self.action == 'PAGE_DOWN' and mry < self.thumb_rect.y:
            entries.set_top(entries.bottom)

        elif self.action == 'PAGE_UP' and mry > self.thumb_rect.y2:
            entries.set_top(entries.top - entries.visible_lines)
        else:
            return False
        return True


class TEXTENSION_OT_suggestions_scrollbar(types.TextOperator):
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        self.entries = entries = get_instance().entries
        line_px = entries.line_height
        span = line_px * len(entries.items)
        self.px_coeff = span / entries.height * 1.015  # Nasty compensation hack
        self.top_org = entries.top
        self.line_height_coeff = 1 / max(1, line_px)
        min_px = min(30, entries.height)
        clamp_ratio = entries.scrollbar.clamp_ratio

        clamp_coeff = (min_px - (min_px / clamp_ratio)) * self.px_coeff
        self.delta_line_coeff = 1 + (clamp_coeff / (entries.max_top * line_px))
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return {'CANCELLED'}

        elif event.type == 'MOUSEMOVE':
            if abs(event.mouse_prev_press_x - event.mouse_x) > 140:
                top = self.top_org
            else:
                dy = event.mouse_prev_press_y - event.mouse_y
                delta_px = (dy * self.px_coeff) * self.delta_line_coeff
                top = self.top_org + delta_px * self.line_height_coeff
            self.entries.set_top(top)
        return {'RUNNING_MODAL'}


def dismiss():
    instance = get_instance()
    if test_and_update(instance, "visible", False):
        TEXTENSION_OT_hit_test.poll(_context)
        _context.region.tag_redraw()
        _context.window.cursor_set('TEXT')
        instance.set_new_hit(None)


def on_insert() -> None:
    """Hook for TEXTENSION_OT_insert"""
    line, col = _context.edit_text.cursor_position
    if _context.edit_text.lines[line].body[col - 1] not in separators:
        bpy.ops.textension.suggestions_complete('INVOKE_DEFAULT')
    else:
        dismiss()


def on_delete() -> None:
    """Hook for TEXTENSION_OT_delete"""
    line, col = _context.edit_text.cursor_position
    lead = _context.edit_text.lines[line].body[:col]
    if not lead.strip() or lead[-1:] in separators | {"."}:
        dismiss()
    elif lead.lstrip():
        instance = get_instance()
        instance.sync_cursor()
        if instance.poll():  # If visible, run completions again.
            bpy.ops.textension.suggestions_complete('INVOKE_DEFAULT')

# import gc
# import tracemalloc

# started = False
class TEXTENSION_OT_suggestions_complete(types.TextOperator):
    keymap("Text Generic", 'SPACE', 'PRESS', ctrl=1)

    @classmethod
    def poll(cls, context):
        return is_spacetext(context.space_data) \
           and is_text(getattr(context, "edit_text", None))

    def execute(self, context):
        text = context.edit_text
        line, col = text.cursor_position
        string = text.as_string()

        # Suggestions should not show up inside comments or multi-line strings
        cursor = text.cursor_sorted
        for t in _iter_expand_tokens(string):
            if cursor in t and  t.type in {'COMMENT', 'MULTILINE_STRING'}:
                return {'CANCELLED'}
        instance = get_instance()
        instance.sync_cursor()

        from jedi.api import Interpreter
        # from .optimizations import Interpreter

        # ret = sorted(interp.complete(line + 1, col, fuzzy=True),
        #     key=lambda x:x.complete is not None, reverse=False)
        interp = Interpreter(string, [])
        ret = interp.complete(line + 1, col)
        bpy.ret = ret
        instance.entries.items = tuple(ret)
        instance.visible = True
        instance.region.tag_redraw()
        return {'CANCELLED'}

    def invoke(self, context, event):
        # Defer completion to the idle pass to make it appear.
        override = dict(_context.copy(), space_data=_context.space_data)
        utils.defer(bpy.ops.textension.suggestions_complete, override)
        return {'FINISHED'}


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
        import queue
        import signal
        import threading

        import _socket
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

        dl_thread = threading.Thread(target=worker_main)
        print("starting worker thread..")
        dl_thread.start()
        return {'FINISHED'}


def iter_instances() -> types.Iterable[Instance]:
    yield from get_instance.__kwdefaults__["cache"].values()



def reset_entries(self, context):
    for instance in iter_instances():
        instance.entries.cache_key = ()
        instance.entries.metrics_key = (0, 0)
        instance.entries.surface_size = (1, 1)
        instance.entries.scrollbar.compute_args = ()
    utils.redraw_editors()


def retself(obj):
    return obj


def set_runtime_uniforms(datapath, attr, value):
    if datapath is not None:
        import operator
        getter = operator.attrgetter(datapath)
    else:
        getter = retself
    for instance in iter_instances():
        getter(instance).update(**{attr: value})
    utils.redraw_editors()


def update_factory(path):
    if "." in path:
        path, attr = path.rsplit(".", 1)
        p_attr = f'{path.rsplit(".", 1)[-1]}_{attr}'
    else:
        attr = p_attr = path
        path = None

    def on_update(self, _, *, data=[path, attr, p_attr]):
        set_runtime_uniforms(data[0], data[1], getattr(self, data[2]))

    d = {"update": on_update, "name": p_attr.replace("_", " ").title()}
    if "color" in p_attr:
        d.update(min=0.0, max=1.0, size=4, subtype='COLOR_GAMMA')
    return d


def update_radius(self, context):
    value = self.radius
    for instance in iter_instances():
        instance.update(radius=value)
        instance.entries.active.update(radius=value)
        instance.entries.hover.update(radius=value)
        instance.entries.scrollbar.update(radius=value)
        instance.entries.scrollbar.thumb.update(radius=value)


def update_resize_highlights(self, context):
    for instance in iter_instances():
        for sizer in instance.resizer.sizers:
            sizer.set_alpha(0.0)


class TEXTENSION_PG_suggestions(bpy.types.PropertyGroup):
    color_kw = {"min": 0, "max": 1, "size": 4, "subtype": 'COLOR_GAMMA'}

    # from bpy.props import IntProperty, BoolProperty, FloatProperty, FloatVectorProperty
    font_size: bpy.props.IntProperty(name="Font Size", update=reset_entries, default=14, min=1, max=144)

    show_resize_highlight: bpy.props.BoolProperty(name="Show Resize Highlights", update=update_resize_highlights, default=True)
    auto_font_size: bpy.props.BoolProperty(name="Automatic Font Size", update=reset_entries, default=True)
    line_padding: bpy.props.FloatProperty(name="Line Height", update=reset_entries, default=1.25, min=0.5, max=4.0)
    text_padding: bpy.props.IntProperty(name="Text Padding", update=reset_entries, default=5, min=0, max=1000)
    scrollbar_width: bpy.props.IntProperty(name="Scrollbar Width", default=18, min=8, max=100)
    radius: bpy.props.FloatProperty(name="Roundness", update=update_radius, default=0.0, min=0.0, max=10.0)
    foreground_color: bpy.props.FloatVectorProperty(
        **color_kw, name="Foreground Color",
        default=(0.4, 0.7, 1.0, 1.0), update=reset_entries
    )
    match_foreground_color: bpy.props.FloatVectorProperty(
        **color_kw, name="Match Foreground Color",
        default=(0.87, 0.6, 0.25, 1.0), update=reset_entries
    )
    background_color: bpy.props.FloatVectorProperty(
        default=(0.15, 0.15, 0.15, 1.0), **update_factory("background_color"))
    border_color: bpy.props.FloatVectorProperty(
        default=(0.3, 0.3, 0.3, 1.0), **update_factory("border_color"))
    border_width: bpy.props.FloatProperty(
        default=1.0, min=0.0, max=20.0, **update_factory("border_width"))

    active_background_color: bpy.props.FloatVectorProperty(
        default=(0.16, 0.22, 0.33, 1.0),
        **update_factory("entries.active.background_color"))
    active_border_color: bpy.props.FloatVectorProperty(
        default=(0.16, 0.29, 0.5, 1.0),
        **update_factory("entries.active.border_color"))
    active_border_width: bpy.props.FloatProperty(
        default=1.0, min=0.0, max=20.0,
        **update_factory("entries.active.border_width"))

    hover_background_color: bpy.props.FloatVectorProperty(
        default=(1.0, 1.0, 1.0, 0.4),
        **update_factory("entries.hover.background_color"))
    hover_border_color: bpy.props.FloatVectorProperty(
        default=(1.0, 1.0, 1.0, 0.2),
        **update_factory("entries.hover.border_color"))
    hover_border_width: bpy.props.FloatProperty(
        default=1.0, min=0.0, max=20.0,
        **update_factory("entries.hover.border_width"))

    thumb_background_color: bpy.props.FloatVectorProperty(
        default=(0.18, 0.18, 0.18, 1.0),
        **update_factory("entries.scrollbar.thumb.background_color"))
    thumb_border_color: bpy.props.FloatVectorProperty(
        default=(0.26, 0.26, 0.26, 1.0),
        **update_factory("entries.scrollbar.thumb.border_color"))
    thumb_border_width: bpy.props.FloatProperty(
        default=1.0, min=0.0, max=20.0,
        **update_factory("entries.scrollbar.thumb.border_width"))


classes = (
    TEXTENSION_PG_suggestions,
    TEXTENSION_OT_suggestions_complete,
    TEXTENSION_OT_suggestions_scrollbar,
    TEXTENSION_OT_suggestions_scroll,
    TEXTENSION_OT_suggestions_navigate,
    TEXTENSION_OT_suggestions_resize,
    TEXTENSION_OT_suggestions_dismiss,
    TEXTENSION_OT_suggestions_commit,
)

_builtins = [
    # "bpy.app",
    # "bpy.app.handlers",
    # "bpy.app.icons",
    # "bpy.app.timers",
    # "mathutils",
    # "mathutils.noise",
    # "builtins",
    # "numpy",
    # "matplotlib"
]

def enable():
    bpy.utils.register_class(TEXTENSION_OT_suggestions_download_jedi)

    # Unless Jedi already exists, it's placed into the directory 'download'.
    # In this case we add it to sys.path to make it globally importable.
    if PLUGIN_PATH not in sys.path:  # TODO: Should be 'download', not root directory.
        sys.path.append(PLUGIN_PATH)

    if not poll_plugin():
        return

    import jedi
    utils.register_class_iter(classes)

    from ... import overrides
    overrides.insert_hooks.append(on_insert)
    overrides.delete_hooks.append(on_delete)

    WidgetBase.prefs = utils.add_settings(TEXTENSION_PG_suggestions)
    jedi.settings.auto_import_modules = set()
    # Include module names which jedi can't infer on its own
    # jedi.settings.auto_import_modules[:] = set(
    #     jedi.settings.auto_import_modules + ["bpy", "_bpy", "numpy", "sys"])

    # Do not let jedi infer anonymous parameters. It's slow and useless.
    jedi.settings.dynamic_params = False

    utils.add_draw_hook(fn=draw, space=SpaceTextEditor, args=())
    from ... import ui
    ui._ht_hooks_window.append(test_suggestions_box)

    from . import resolvers, patches
    resolvers.build_context_resolver()
    patches.apply()

    from .optimizations import setup2
    setup2()
        

def poll_plugin(*, _poll_cache=[False, 0]):
    """Return whether parso/jedi are importable, without importing them."""
    result, next_check = _poll_cache
    check_time = time.monotonic()
    if not result and next_check < check_time:
        from importlib.util import find_spec
        result = bool(find_spec("parso") and find_spec("jedi"))
        _poll_cache[:] = (result, check_time + 2.0)
    return result
         

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
    else:
        suggestions = prefs.suggestions
        layout.prop(suggestions, "auto_font_size")
        row = layout.row()
        row.prop(suggestions, "font_size")
        row.enabled = not suggestions.auto_font_size

        # layout.prop(suggestions, "contrast")
        layout.prop(suggestions, "show_resize_highlight")
        layout.prop(suggestions, "line_padding")
        layout.prop(suggestions, "text_padding")
        layout.separator(factor=3)
        layout.prop(suggestions, "radius", slider=True)
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
        layout.prop(suggestions, "thumb_background_color")
        layout.prop(suggestions, "thumb_border_color")
        layout.prop(suggestions, "thumb_border_width")


def disable():

    from ... import overrides
    overrides.insert_hooks.remove(on_insert)
    overrides.delete_hooks.remove(on_delete)

    utils.unregister_class(TEXTENSION_OT_suggestions_download_jedi)
    utils.remove_settings(TEXTENSION_PG_suggestions)
    ui._ht_hooks_window.remove(test_suggestions_box)
    utils.remove_draw_hook(draw)
    utils.unregister_class_iter(reversed(classes))
    get_instance.__kwdefaults__["cache"].clear()
