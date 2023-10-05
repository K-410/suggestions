"""This module implements word hover."""

from textension import ui, utils

from . import settings, _setup_jedi, _context

import bpy
import blf

from bpy.app.timers import (
    register as register_timer,
    unregister as unregister_timer,
    is_registered as is_timer_registered
)


# Hit test the current word, or find a word on a timer.
def hover_handler(x: int, y: int) -> None:
    if is_timer_registered(find_word_and_show):
        unregister_timer(find_word_and_show)

    hover = get_hover_from_space(_context.space_data)

    if hover.is_visible and hover.hit_test_word(x, y):
        return hover.word

    # Timer callbacks lose all context. We just need to store the area.
    # The coords are stored just for convenience.
    hover.coord = x, y
    hover.runtime.area  = _context.area

    register_timer(find_word_and_show, first_interval=0.3)


@utils.inline
def get_dna_cursor(text):
    return utils.operator.attrgetter("curl.contents", "sell.contents", "curc", "selc")


@utils.inline
def cursor_set(x: int, y: int):
    return utils.partial(utils._call, "TEXT_OT_cursor_set", {})


# TODO: Should also be used by overrides.default module.
def set_cursor_with_offset(x, y):
    from textension.overrides.default import restore_offset
    with (ctx := restore_offset()):
        try:
            cursor_set({'x': x, 'y': y - ctx.result - 2})

        # Happens if the operator is called while inside a call.
        except RuntimeError:
            return None
    return None


# Called by the timer to do hit testing and show the hover.
def find_word_and_show():
    # Can be None. We can't guarantee the area exists anymore.
    area = utils.validate_area(Hover.runtime.area)
    try:
        space = area.spaces.active
        text  = space.text
        dna   = text.internal
    except AttributeError:
        return None

    context_dict = dict(
        window=utils.window_from_area(area),
        area=area,
        region=area.regions[-1],
        space_data=space,
        edit_text=text)

    hover = get_hover_from_space(space)
    curl, sell, curc, selc = get_dna_cursor(dna)

    with utils.context_override(**context_dict):
        set_cursor_with_offset(*hover.coord)
        line, column = text.cursor_focus

    # Don't use RNA to avoid redraws. This is just word hit testing.
    dna.curl.contents = curl
    dna.sell.contents = sell
    dna.curc = curc
    dna.selc = selc

    if not settings.loaded:
        _setup_jedi(force=True)

    from .patches.common import interpreter
    interpreter.set_session(text)

    s, start, end = get_word_at(text.lines[line].body, column)

    # Only valid names, not numbers, separators or anything else.
    if not s.isidentifier():
        return None

    names = interpreter.goto(line + 1, start)
    if not names:
        return None

    word = hover.word
    space = hover.space_data

    # "The actual line height hurr durr". Thanks Blender. dyec?
    not_line_height = space.runtime._lheight_px
    line_height = int(not_line_height * 1.3)

    if start == end:
        return None

    string = text.lines[line].body[start:end]
    word_x, word_y = space.region_location_from_cursor(line, start)

    # Fix for ``region_location_from_cursor``.
    word_y += line_height - not_line_height

    word.rect.position = word_x, word_y + space.offsets.y

    # ``blf.dimensions`` works on glyph dimensions, not fixed sizes.
    word_width = max(space.cwidth * len(string), int(blf.dimensions(1, string)[0]))
    word.rect.size = word_width, line_height
    hover.word.cache_key = space.top, space.font_size, text.cursor_focus

    with utils.context_override(**context_dict):
            return None
        ui.set_hit(word)

    hover.set_from_string(names[0].type)
    hover.fit()

    region = utils.region_from_space_data(space, 'WINDOW')
    if word_y + line_height + hover.rect.height > region.height:
        word_y -= line_height
    else:
        word_y += line_height

    # Pop-up position.
    hover.rect.position = word_x, word_y + space.offsets.y


def get_word_at(string, column):
    import re

    left_string = string[:column]
    right_string = string[column:]
    left = re.match(r".*?(\w+)\.?$", left_string)
    right = re.match(r"^\.?(\w+).*$", right_string)

    start = end = column
    if left:
        start, end = left.span(1)
        if right:
            if left_string.endswith("."):
                start = column + right.start(1)
                end = column + right.end(1)

            elif not right_string.startswith("."):
                start = left.start(1)
                end = column + right.end(1)

    return string[start:end], start, end


class Word(ui.widgets.Widget):
    """For hit testing the word of a visible hover."""

    background_color = 0.0, 0.0, 0.0, 0.0
    border_color     = 0.0, 0.0, 0.0, 0.0
    border_width     = 1.0

    cursor = 'TEXT'
    cache_key = ()

    def __init__(self, hover):
        super().__init__(None)
        self.hover = hover

    def on_enter(self):
        self.hover.is_visible = True

    def on_leave(self):
        self.hover.is_visible = False

    def on_activate(self) -> None:
        return ui.utils.PASS_THROUGH


class Hover(ui.widgets.Popup):
    """The hover pop-up."""

    space_data: bpy.types.SpaceTextEditor
    word:  Word

    runtime = utils.namespace(area=None)

    is_visible = False  # Whether the hover is shown.
    font_id    = 1

    coord     = (-1, -1)
    text      = utils._forwarder("space_data.text")
    font_size = utils._forwarder("space_data.font_size")
    key       = utils._forwarder("space_data.top", "font_size", "text.cursor_focus")

    def draw(self):
        if self.key == self.word.cache_key:
            super().draw()
        else:
            self.is_visible = False

    def __init__(self, st):
        self.space_data = st
        self.word = Word(self)
        super().__init__(None)

    def hit_test_word(self, x, y):
        if self.text and self.word.hit_test(x, y):
            return True
        self.is_visible = False
        utils.safe_redraw_from_space(self.space_data)
        return False


@utils.inline
def get_hover_instance() -> Hover:
    return utils.make_space_data_instancer(Hover)


@utils.inline
def get_hover_from_space(space) -> Hover:
    return get_hover_instance.from_space


def draw_hover():
    hover = get_hover_instance()
    if hover.is_visible:
        hover.draw()


def disable():
    if is_timer_registered(find_word_and_show):
        unregister_timer(find_word_and_show)
    ui.remove_hit_test(hover_handler)
    ui.remove_draw_hook(draw_hover)


def enable():
    ui.add_draw_hook(draw_hover, draw_index=12)
    ui.add_hit_test(hover_handler, 'TEXT_EDITOR', 'WINDOW')
