import os
import sys
import time

import blf
import bpy
from bpy.types import SpaceTextEditor

from ... import TEXTENSION_OT_hit_test, gl, types, utils, _iter_expand_tokens
from ...types import is_spacetext, is_text

# TODO: Testing
from dev_utils import enable_breakpoint_hook, per, measure, total, accumulate
# import threading, debugpy
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


class Widget(gl.GLRoundedRect):
    cursor: str
    def enter(self): return None
    def leave(self): return None
    def activate(self): return None
    def __init__(self, *colors):
        super().__init__(*colors)


class Scrollbar(Widget):
    action: str = 'undefined'
    clamp_ratio: float = 0.0
    compute_args: tuple[int, int, int, float] = (0, 0, 0, 0.0)
    compute_cache: tuple[int, int] = (0, 0)

    def __init__(self, parent: "ListBox"):
        super().__init__(0.18, 0.18, 0.18, 0.0)
        self.parent = parent
        self.thumb = Widget(0.27, 0.27, 0.27, 1.0)
        self.thumb.activate = lambda: bpy.ops.textension.suggestions_scrollbar('INVOKE_DEFAULT')

    def activate(self):
        bpy.ops.textension.suggestions_scroll('INVOKE_DEFAULT', action=self.action)

    def hit_test(self, mrx, mry):
        if super().hit_test(mrx, mry):
            self.action = 'PAGE_DOWN' if mry < self.thumb.y else 'PAGE_UP'
            if self.thumb.hit_test(mrx, mry):
                return self.thumb
            return self

    def draw(self):
        y, h = self.compute_geometry()
        if h != 0:
            parent = self.parent
            w = parent.scrollbar_width
            x = parent.parent.x + parent.width - w
            y += parent.parent.y
            self(x, parent.parent.y + 1, w, parent.height - 2)
            self.thumb(x, y, w, h)

    def compute_geometry(self):
        """Compute the vertical position of the scrollbar thumb."""
        parent = self.parent
        compute_args = (parent.height, parent.line_height, len(parent.items), parent.top)

        if test_and_update(self, "compute_args", compute_args):
            height, line_height, nlines, top = compute_args
            visible_lines = height / line_height
            if visible_lines >= nlines:
                self.clamp_ratio = 0.0
                self.geometry_cache = (0, 0)
            else:
                # Minimum thumb height before it's clamped to keep it clickable.
                min_ratio = min(30, height) / height
                ratio = visible_lines / nlines

                if ratio > min_ratio:
                    y = int(height * (1 - (top + visible_lines) / nlines))
                    h = int((height * (1 - top / nlines)) - y)
                    self.clamp_ratio = 1.0
                else:
                    ymax = height - int((height * (1 - (min_ratio - ratio))) * top / nlines)
                    y = int(ymax - height * min_ratio)
                    h = ymax - int(ymax - height * min_ratio)
                    # The height is clamped. The ratio represents the size difference.
                    self.clamp_ratio = min_ratio / ratio
                self.geometry_cache = (max(0, y), h)
        return self.geometry_cache


class Resizer(Widget):
    def __init__(self, parent: 'Instance', cursor: str, action: str):
        super().__init__(0.5, 0.5, 0.5, 0.0)
        self.parent = parent
        self.action = action
        self.cursor = cursor
        self.sizers = [self]
        self.enter = lambda: self.set_alpha(1.0)
        self.leave = lambda: self.set_alpha(0.0)
        self.activate = lambda action=action: \
            bpy.ops.textension.suggestions_resize('INVOKE_DEFAULT', action=action)

    def set_alpha(self, value):
        for resizer in self.sizers:
            if test_and_update(resizer.background, "w", value):
                self.parent.region.tag_redraw()


class BoxResizer(Resizer):
    def __init__(self, parent: "Instance"):
        super().__init__(parent, 'SCROLL_XY', 'CORNER')
        self.sizers[:] = (self.horz, self.vert) = \
            Resizer(parent, 'MOVE_X', 'HORIZONTAL'), \
            Resizer(parent, 'MOVE_Y', 'VERTICAL')

    def hit_test(self, mrx, mry):
        if mrx >= self.parent.x2 - 8 and mry <= self.parent.y + 8:
            return self
        for s in self.sizers:
            if s.hit_test(mrx, mry):
                return s
        return None

    def draw(self, x, y, w, h):
        self.horz(x + w - 4, y, 5, h)
        self.vert(x, y - 1, w, 5)


class EntryRect(gl.GLRoundedRect):
    index: int

    def __init__(self, *color, index=0):
        super().__init__(*color)
        self.index = index


class ListBox(Widget):
    cache_key: tuple = ()
    scroll: Scrollbar

    line_heightf: float  = 1.0
    line_height: int     = 1         # Line height
    font_size: int       = 14
    text_padding: int    = 5
    line_padding: float  = 1.45
    scrollbar_width: int = 20

    metrics_key: tuple[int, int] = (0, 0)

    def __init__(self, parent: "Instance"):
        super().__init__()
        self.parent = parent
        self.desc   = 0
        self.hash   = 0
        self.items  = ()        # Completions
        self.top    = 0.0       # Scroll position

        # Compensate for the top/bottom 1px border of the box.
        self.surface_size = self.width, self.height = 300, 200
        self.surface    = gl.GLTexture(self.width, self.height - 2)

        self.scroll     = Scrollbar(self)
        self.selection  = EntryRect(0.3, 0.4, 0.8, 0.4)
        self.hover      = EntryRect(1.0, 1.0, 1.0, 0.08, index=-1)

    def leave(self):
        if test_and_update(self.hover, "index", -1):
            self.parent.region.tag_redraw()

    def activate(self):
        bpy.ops.textension.suggestions_commit(
            'INVOKE_DEFAULT', index=self.hover.index)

    def draw(self):
        parent = self.parent
        blf.color(1, 0.4, 0.7, 1.0, 1)

        # There's no scissor/clip as of 3.2, so we draw to an off-screen
        # surface instead. The surface is cached for performance reasons.
        size = (w, h) = (self.width, self.height)
        if test_and_update(self, "surface_size", size):
            self.surface.resize(w, h - 2)

        # If the items changed, reset scroll, selection and hover
        if test_and_update(self, "hash", hash(self.items)):
            self.selection.index = 0
            self.hover.index = -1  # Hover is visible only on cursor hover.
            self.top = 0.0

        # If the font metrics changed, re-compute line height
        # At 1.77 scale, dpi is halved and pixel_size is doubled. Don't ask why.
        metrics_key = self.font_size, int(system.dpi * system.pixel_size)
        blf.size(1, *metrics_key)
        if test_and_update(self, "metrics_key", metrics_key):
            _, adh = blf.dimensions(1, "Ag")
            self.line_heightf = adh * self.line_padding
            self.line_height = int(self.line_heightf)

            _, xh = blf.dimensions(1, "x")
            self.desc = (adh - xh) * 0.5  # Averaged ascender/descender

        lh = self.line_height
        top = self.top
        # The offset in pixels into the current top index
        offset = top % 1 * self.line_heightf
        top_int = int(top)
        if test_and_update(self, "cache_key", (top, self.hash, lh, size)):
            max_items = int(h / self.line_heightf + 2)
            text_x = self.text_padding * system.wu * 0.05
            text_y = h + self.desc - lh + offset + (3 * system.wu * 0.05)

            with self.surface.bind():
                for item in self.items[top_int:top_int + max_items]:
                    blf.position(1, text_x, text_y, 0)
                    blf.draw(1, item.name)
                    text_y -= lh

        x = parent.x
        y = parent.y
        self.surface(x, y + 1, w, h - 2)

        for rect in (self.selection, self.hover):
            height = lh
            ypos = y + h - height - (height * (rect.index - top_int)) + offset
            if ypos <= y:  # Selection y is below box
                height -= (y - ypos) + 1
                ypos = y + 1
            if ypos + height >= y + h:  # Selection y + height is above box
                height -= (ypos + height - (y + h)) + 1
            rect(x + 1, ypos, w - 2, height)

        self.scroll.draw()

    def hit_test(self, mrx, mry):
        if hit := self.scroll.hit_test(mrx, mry):
            return hit

        hit_index = int(self.top + ((self.parent.y2 - mry) / self.line_height))
        if hit_index < len(self.items):
            if test_and_update(self.hover, "index", hit_index):
                self.parent.region.tag_redraw()
            return self
        return None

    def set_top(self, top: float) -> None:
        """Assign a new top value."""
        if test_and_update(self, "top", max(0, min(self.max_top, top))):
            self.parent.region.tag_redraw()

    @property
    def max_top(self):
        return len(self.items) - (self.height / self.line_height)


class Instance(gl.GLRoundedRect):
    def __init__(self, st: SpaceTextEditor) -> None :
        super().__init__(0.15, 0.15, 0.15, 1.0)
        self.set_border_color(0.3, 0.3, 0.3, 1.0)

        self.x = 0
        self.y = 0
        self.hit = None
        self.visible = False
        self.cursor_position = (0, 0)
        self.region: bpy.types.Region = utils.region_from_space_data(st)
        self.entries = ListBox(self)
        self.resizer = BoxResizer(self)

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
        if not super().hit_test(mrx, mry):
            return None

        # Test if we're inside the resize handles
        if hit := self.resizer.hit_test(mrx, mry):
            _context.window.cursor_set(hit.cursor)
        else:
            # Test if we're hitting any entry
            _context.window.cursor_set("DEFAULT")
            if hit := self.entries.hit_test(mrx, mry):
                pass

            else:  # The box was hit, but none of the widgets were.
                return types.noop

        self.set_new_hit(hit)
        return hit.activate

    def set_new_hit(self, hit=None):
        if self.hit is not hit:
            if self.hit is not None:
                self.hit.leave()
            self.hit = hit
            if hit is not None:
                hit.enter()

    def draw(self):
        entries = self.entries
        st = _context.space_data
        w, h = entries.width, entries.height
        x, y = st.region_location_from_cursor(*st.text.cursor_position)
        # Align the box to below the cursor.
        y -= h - st.offsets[1] - round(4 * system.wu * 0.05)
        self.x = x
        self.y = y

        self(x, y, w, h)
        entries.draw()
        self.resizer.draw(x, y, w, h)


def get_instance(*, cache={}, context=_context) -> Instance:
    try:
        return cache[context.space_data]
    except KeyError:
        st = context.space_data
        assert is_spacetext(st), "Not SpaceTextEditor, got %s" % st
        return cache.setdefault(st, Instance(st))


def clear_instances_cache() -> None:
    """Clear suggestions box instances cache."""
    get_instance.__kwdefaults__["cache"].clear()


def test_suggestions_box(data: types.HitTestData) -> types.Callable | None:
    """Hit test hook for TEXTENSION_OT_hit_test."""
    instance = get_instance()
    if instance.poll():
        ret = instance.hit_test(*data.pos)
        # If a previous hit exists, call its leave handler.
        if ret in {types.noop, None}:
            instance.set_new_hit(None)
        return ret
    return None


def draw():
    """Draw callback for suggestions box."""
    instance = get_instance()
    if not instance.poll():
        return
    instance.draw()


class TEXTENSION_OT_suggestions_commit(types.TextOperator):
    bl_options = {'INTERNAL'}
    index: bpy.props.IntProperty(default=-1, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return get_instance().poll()

    def execute(self, context):
        instance = get_instance()
        entries = instance.entries
        is_return = False
        # When self.index is -1 or not given, use the currently active index.
        if self.index == -1:
            assert entries.selection.index < len(entries.items)
            self.index = entries.selection.index
            is_return = True

        complete_str = entries.items[self.index].complete
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
        self.entries = get_instance().entries
        self.width, self.height = self.entries.surface.size
        self.min_height = self.entries.line_height
        self.min_width = int(150 * system.wu * 0.05)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return {'CANCELLED'}

        elif event.type == 'MOUSEMOVE':
            entries = self.entries
            x_delta = event.mouse_x - event.mouse_prev_press_x
            y_delta = event.mouse_y - event.mouse_prev_press_y

            *size, = self.entries.surface.size
            if self.action in {'CORNER', 'HORIZONTAL'}:
                size[0] = max(self.min_width, self.width + x_delta)

            if self.action in {'CORNER', 'VERTICAL'}:
                size[1] = max(self.min_height, self.height - y_delta)

                # Resizing vertically past the bottom entry moves the top up.
                if (top := entries.top) > 0:
                    view = (size[1] / entries.line_height)
                    if (bottom := top + view) > len(entries.items):
                        entries.set_top(max(0, top - (bottom - len(entries.items))))

            (entries.width, entries.height) = size
            context.region.tag_redraw()
        return {'RUNNING_MODAL'}



class TEXTENSION_OT_suggestions_navigate(types.TextOperator):
    bl_options = {'INTERNAL'}

    action: bpy.props.EnumProperty(
        items=[(v, "", "") for v in ("DOWN", "UP", "PAGE_DOWN", "PAGE_UP")])

    @classmethod
    def poll(cls, context):
        return is_spacetext(context.space_data) and get_instance().poll()

    def execute(self, context):
        instance = get_instance()
        entries = instance.entries
        # Page is visible lines minus one. Makes it easier to visually track.
        page = int(entries.height / entries.line_height) - 1

        if self.action in {'DOWN', 'UP'}:

            # When only a single item is shown, up/down keys are passed on
            # and the box is closed. We also invalidate the cursor position
            # so that moving back doesn't re-show the box.
            if len(entries.items) == 1:
                instance.cursor_position = -1, -1
                return {'PASS_THROUGH'}

            value = entries.selection.index + (1 if self.action == 'DOWN' else -1)
            new_index = value % len(entries.items)
        elif self.action == 'PAGE_DOWN':
            new_index = min(len(entries.items) - 1, entries.selection.index + page)
        else:
            new_index = max(0, entries.selection.index - page)

        if new_index >= entries.top + page:  # New index is below bottom
            entries.top = new_index - page
        elif new_index < entries.top:        # New index is above top
            entries.top = new_index

        entries.selection.index = new_index
        context.region.tag_redraw()
        return {'FINISHED'}

    @classmethod
    def register_keymaps(cls):
        from ...km_utils import kmi_new
        kmi_new(cls, "Text Generic", cls.bl_idname, 'UP_ARROW', 'PRESS', repeat=True).action = 'UP'
        kmi_new(cls, "Text Generic", cls.bl_idname, 'DOWN_ARROW', 'PRESS', repeat=True).action = 'DOWN'

        for action in ('PAGE_DOWN', 'PAGE_UP'):
            kmi_new(cls, "Text Generic", cls.bl_idname, action, 'PRESS', repeat=True).action = action


class TEXTENSION_OT_suggestions_scroll(types.TextOperator):
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
        self.page = entries.height / entries.line_height
        self.thumb = entries.scroll.thumb

        # Operator was invoked from clicking the gutter.
        if event.type == 'LEFTMOUSE':
            self.action = 'PAGE_DOWN'

            if event.mouse_region_y >= self.thumb.y:
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
        top = self.entries.top
        if self.action == 'PAGE_DOWN' and mry < self.thumb.y:
            self.entries.set_top(top + self.page)

        elif self.action == 'PAGE_UP' and mry > self.thumb.y + self.thumb.height:
            self.entries.set_top(top - self.page)
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
        self.entries = entries = get_instance().entries
        line_px = entries.line_height
        span = line_px * len(entries.items)
        self.px_coeff = span / entries.height * 1.015  # Nasty compensation hack
        self.top_org = entries.top
        self.line_height_coeff = 1 / max(1, line_px)
        min_px = min(30, entries.height)
        clamp_ratio = entries.scroll.clamp_ratio

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
        return


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

import gc
import tracemalloc

started = False
class TEXTENSION_OT_suggestions_complete(types.TextOperator):
    @classmethod
    def poll(cls, context):
        return is_spacetext(context.space_data) \
           and is_text(getattr(context, "edit_text", None))

    def execute(self, context):
        text = context.edit_text
        line, col = text.cursor_position
        string = text.as_string()

        # Suggestions should not show up inside comments
        cursor = text.cursor_sorted
        for t in _iter_expand_tokens(string):
            if cursor in t and  t.type == 'COMMENT':
                return {'CANCELLED'}

        instance = get_instance()
        instance.sync_cursor()
        # TODO: If optimized, use that version.
        from jedi.api import Interpreter
        # from .optimizations import Interpreter2 as Interpreter, _clear_caches
        # from .optimizations import Interpreter2 as Interpreter

        # ret = sorted(interp.complete(line + 1, col, fuzzy=True),
        #     key=lambda x:x.complete is not None, reverse=False)
        # with measure:
        # tracemalloc.start(10)
        interp = Interpreter(string, [])
        # ret = interp.complete_unsafe(line + 1, col)
        ret = interp.complete(line + 1, col)
        bpy.ret = ret
        instance.entries.items = tuple(ret)

        # instance.items = tuple(entry.name for entry in ret)
        # inference_state = interp._inference_state
        # inference_state.analysis.clear()
        # inference_state.compiled_cache.clear()
        # inference_state.memoize_cache.clear()
        # inference_state.inferred_element_counts.clear()
        # inference_state.mixed_cache.clear()
        # inference_state.access_cache.clear()
        # inference_state.module_cache._name_cache.clear()
        # _clear_caches()
        # del ret, interp, inference_state
        # from jedi.cache import clear_time_caches
        # clear_time_caches(True)
        # del clear_time_caches
        # gc.collect()

        # bpy.snapshot = snapshot = tracemalloc.take_snapshot()
        # tracemalloc.stop()
        
        # top_stats = snapshot.statistics('lineno')
        # os.system("cls")
        # for stat in top_stats[:10]:
        #     print(stat)

        # try:
        #     last = bpy.last
        # except:
        #     last = bpy.last = len(gc.get_objects())
        # else:
        #     objects = gc.get_objects()
        #     bpy.last = curr = len(objects)
        #     print(curr, last, curr - last, "new")
        #     if curr - last > 0:
        #         print("last few objects:", [type(o) for o in objects[-min(curr - last, 5):]])


        instance.visible = True
        TEXTENSION_OT_hit_test.poll(_context)
        instance.region.tag_redraw()
        return {'CANCELLED'}

    def invoke(self, context, event):
        # Defer completion to the idle pass to make it appear.
        override = dict(_context.copy(), space_data=_context.space_data)
        utils.defer(bpy.ops.textension.suggestions_complete, override)
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

        # bpy.app.timers.register(timer_main)
        dl_thread = threading.Thread(target=worker_main)
        print("starting worker thread..")
        dl_thread.start()
        return {'FINISHED'}


def update_factory(cls, attr, after_update=types.noop):
    def on_update(self, _, *, cls=cls, attr=attr):
        setattr(cls, attr, getattr(self, attr))
        after_update()
    return on_update


def invalidate_surfaces():
    """Invalidate surface so must be redrawn the next time"""
    for instance in get_instance.__kwdefaults__["cache"]:
        instance.entries.cache_key = ()


class TEXTENSION_PG_suggestions_settings(bpy.types.PropertyGroup):
    font_size: bpy.props.IntProperty(
        name="Font Size", description="Font size for suggestions entries",
        update=update_factory(ListBox, "font_size"),
        default=ListBox.font_size, min=1, max=144)

    line_padding: bpy.props.FloatProperty(
        name="Line Padding", description="Line Padding",
        update=update_factory(ListBox, "line_padding"),
        default=ListBox.line_padding, min=0.0, max=4.0)

    text_padding: bpy.props.IntProperty(
        name="Text Padding", description="Text Padding",
        update=update_factory(ListBox, "text_padding", invalidate_surfaces),
        default=ListBox.text_padding, min=0, max=1000)

    scrollbar_width: bpy.props.IntProperty(
        name="Scrollbar Width", description="Scrollbar Width",
        update=update_factory(ListBox, "scrollbar_width"),
        default=ListBox.scrollbar_width, min=8, max=100)


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
    utils.register_class_iter(classes)

    from ... import TEXTENSION_OT_insert, TEXTENSION_OT_delete
    from ...utils import TextensionPreferences
    from . import resolvers

    TEXTENSION_OT_insert.insert_hooks.append(on_insert)
    TEXTENSION_OT_delete.delete_hooks.append(on_delete)
    TextensionPreferences.suggestions = bpy.props.PointerProperty(type=TEXTENSION_PG_suggestions_settings)

    # Include module names which jedi can't infer on its own
    jedi.settings.auto_import_modules[:] = set(
        jedi.settings.auto_import_modules + ["bpy", "_bpy", "numpy", "sys"])

    utils.add_draw_hook(fn=draw, space=SpaceTextEditor, args=())
    utils.add_hittest(test_suggestions_box)

    resolvers.set_bpy_struct_mapping()
    resolvers.patch_get_builtin_module_names()
    resolvers.patch_inference_state_process()
    resolvers.patch_anonymous_param()
    resolvers.patch_getattr_paths()
    from .optimizations import setup2
    # setup2()


_poll_cache = [False, 0]


def poll_plugin():
    """Return whether parso/jedi are importable, without importing them."""
    poll, timeout = _poll_cache
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

def print_traceback():  # XXX: For development
    import traceback
    print(traceback.format_exc())

def disable():
    bpy.utils.unregister_class(TEXTENSION_OT_suggestions_download_jedi)

    from ... import TEXTENSION_OT_insert, TEXTENSION_OT_delete
    from ...utils import TextensionPreferences
    del TextensionPreferences.suggestions

    TEXTENSION_OT_insert.insert_hooks.remove(on_insert)
    TEXTENSION_OT_delete.delete_hooks.remove(on_delete)

    utils.remove_hittest(test_suggestions_box)
    utils.remove_draw_hook(draw)
    clear_instances_cache()
    from . import resolvers
    resolvers.RnaResolver.__new__.__kwdefaults__["cache"].clear()
    resolvers.patch_inference_state_process(restore=True)
    utils.unregister_class_iter(reversed(classes))