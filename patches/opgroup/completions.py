from textension.utils import _named_index, _forwarder, Aggregation, lazy_class_overwrite, factory, inline
from functools import partial
from operator import methodcaller, attrgetter, eq
from jedi.api import classes
from ..tools import state
from ... import settings


def apply():
    optimize_Completion_complete()


@inline
def slice_to(length: int) -> slice:
    return partial(slice, None)


@inline
def map_eq(sequence1, sequence2) -> map:
    return partial(map, eq)


@inline
def map_lower(sequence) -> map:
    return partial(map, str.lower)


@inline
def map_strings(sequence) -> map:
    return partial(map, attrgetter("string_name"))


class CompletionBase(Aggregation, classes.Completion):
    _like_name: str
    _inference_state = state
    _cached_name = None

    string_name            = _named_index(0)
    _string_name_lower     = _named_index(1)
    _same_name_completions = _named_index(2)

    get_name_at = _forwarder("_same_name_completions.__getitem__")
    _name = property(methodcaller("get_name_at", 0))
    __hash__ = object.__hash__

    @lazy_class_overwrite
    def _like_name_length(self):
        return len(self._like_name)

    # Returns the like based on case sensitivity for tests.
    @lazy_class_overwrite
    def test_like_name(self):
        if settings.use_case_sensitive_search:
            return self._like_name
        return self._like_name.lower()

    _is_fuzzy   = property(partial(type(settings).use_fuzzy_search.__get__, settings))
    _is_ordered = property(partial(type(settings).use_ordered_fuzzy_search.__get__, settings))


@factory
def sort_completions(completions):
    from itertools import compress, count, repeat
    from builtins import next
    import operator
    from .completions import settings

    get_lower = operator.attrgetter("_string_name_lower")
    single = repeat("_")
    double = repeat("__")

    @inline
    def map_startswith(iterable1, iterable2):
        return partial(map, str.startswith)

    @inline
    def map_not(iterable) -> map:
        return partial(map, operator.not_)
    
    @inline
    def map_get_lower(completions: list[CompletionBase]) -> map:
        return partial(map, operator.itemgetter(1))

    @inline
    def sorted_lower(strings) -> list:
        return partial(sorted, key=get_lower)

    @inline
    def find(string, substr):
        return str.find

    def fuzzy_sort(names, like_name):
        test_sequence = tuple(enumerate(like_name))
        names = list(names)
        weights = {}
        like_name_lower = like_name.lower()
        like_name_len = len(like_name)

        for name in names:
            rank = 0
            index = 0
            string = name.string_name
            string_lower = name._string_name_lower
            length_bias = like_name_len - len(string)

            # Long names or names that don't strictly match.
            if length_bias < -20 or like_name_lower not in string_lower:
                rank = -500

            else:
                if string.startswith(like_name):
                    rank = 1000

                start = 0
                for index, char in test_sequence:
                    test = find(string_lower, char, start)
                    if test is not -1:
                        # Relative locality + case match + index match.
                        rank += (start - test) + (string[test] is char) + (test is index)
                        start = test
                    else:
                        rank -= 10
                        start = index
            weights[name] = (rank, length_bias, string[index:])
        return sorted(names, key=weights.__getitem__, reverse=True)


    def sort_completions(names: list[CompletionBase], like_name: str):
        if settings.use_fuzzy_search and like_name:
            names = fuzzy_sort(names, like_name)

        else:
            # Names starting with underscores will appear first.
            names = sorted_lower(names)

            # Find the first name without an underscore prefix.
            select = map_not(map_startswith(map_get_lower(names), single))
            if start := next(compress(count(), select), None):
                underscores = names[:start]

                # Find the first name without double underscore prefix.
                select = map_not(map_startswith(map_get_lower(underscores), double))
                if start_single := next(compress(count(), select), None):
                    underscores = underscores[start_single:] + underscores[:start_single]

                # [names] + [single underscore] + [double underscore]
                names = names[start:] + underscores
        return names
    return sort_completions


def convert_completions(completions: list[classes.Completion], like_name):
    """Convert completions from jedi's to our own."""

    class Completion(CompletionBase):
        _like_name = like_name
        _stack     = None

    ret = []
    for comp in completions:
        name = comp.name
        ret += Completion((name, name.lower(), [comp._name])),
    return ret


def optimize_Completion_complete():
    from jedi.api.completion import (
        Completion, _extract_string_while_in_string, complete_dict, complete_file_name)
    from .completions import sort_completions, filter_names, convert_completions

    def complete(self: Completion):
        leaf = self._module_node.get_leaf_for_position(self._original_position, include_prefixes=True)
        # TODO: Jedi should extract strings only if we're inside a string.
        string, start_leaf, quote = _extract_string_while_in_string(leaf, self._original_position)

        if string is not None:
            prefixed_completions = complete_dict(
                self._module_context,
                self._code_lines,
                start_leaf or leaf,
                self._original_position,
                None if string is None else quote + string,
                fuzzy=self._fuzzy,
            )
            if not prefixed_completions:
                prefixed_completions = list(complete_file_name(
                    self._inference_state, self._module_context, start_leaf, quote, string,
                    self._like_name, self._signatures_callback,
                    self._code_lines, self._original_position,
                    self._fuzzy
                ))
            if not prefixed_completions and "\n" in string:
                prefixed_completions = self._complete_in_string(start_leaf, string)
            return convert_completions(prefixed_completions, self._like_name)

        completion_names = self._complete_python(leaf)[1]
        completions = filter_names(completion_names, self.stack, self._like_name)
        return sort_completions(completions, self._like_name)

    Completion.complete = complete


@inline
def contains_fuzzy_unordered(query: str, string: str) -> bool:

    @inline
    def replace(s):
        return str.replace

    def contains_fuzzy_unordered(query: str, string: str):
        for char in query:
            if char in string:
                string = replace(string, char, "", 1)
                continue
            return False
        return True
    return contains_fuzzy_unordered


@inline
def contains_fuzzy_ordered(query: str, string: str):

    @inline
    def get_index(string, char):
        return str.index

    def contains_fuzzy_ordered(query: str, string: str):
        index = -1
        try:
            for char in query:
                index = get_index(string, char, index + 1)
                continue
        except:
            return False
        return True
    return contains_fuzzy_ordered


@inline
def filter_names(completions, stack, like_name):
    from textension.utils import defaultdict_list
    from .completions import contains_fuzzy_unordered
    from itertools import compress, repeat
    from operator import itemgetter
    from builtins import map, zip

    from .completions import CompletionBase, map_strings, map_lower, map_eq, slice_to

    search_types = (partial(map, contains_fuzzy_unordered),
                    partial(map, contains_fuzzy_ordered))

    def filter_names(completions, stack, like_name):

        class Completion(CompletionBase):
            _like_name = like_name
            _stack     = stack

        strings = map_strings(completions)

        # If ``like_name`` is not empty, it means we need to do matching.
        if like_name:
            if not settings.use_case_sensitive_search:
                strings = map_lower(strings)
                like_name = Completion.test_like_name

            if settings.use_fuzzy_search:
                map_fuzzy_func = search_types[settings.use_ordered_fuzzy_search]
                strings = map_fuzzy_func(repeat(like_name), strings)

            else:
                strings = map(itemgetter(slice_to(Completion._like_name_length)), strings)
                strings = map_eq(repeat(like_name), strings)

        names = defaultdict_list()
        for name in compress(completions, strings):
            names[name.string_name] += name,

        return map(Completion, zip(names, map_lower(names), names.values()))
    return filter_names
