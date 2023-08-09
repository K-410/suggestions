from textension.utils import _patch_function, factory

def apply():
    optimize_filter_names()
    optimize_Completion_complete()


@factory
def _sort_completions(completions):
    from itertools import compress, count, repeat
    from builtins import map, next, sorted
    from operator import attrgetter, not_

    get_lower = attrgetter("_string_name_lower")
    startswith = str.startswith
    single = repeat("_")
    double = repeat("__")

    def _sort_completions(names):
        # Names starting with underscores will appear first.
        names = sorted(names, key=get_lower)

        # Find the first name without an underscore prefix.
        select = map(not_, map(startswith, map(get_lower, names), single))
        if start := next(compress(count(), select), None):
            underscores = names[:start]

            # Find the first name without double underscore prefix.
            select = map(not_, map(startswith, map(get_lower, underscores), double))
            if start_single := next(compress(count(), select), None):
                underscores = underscores[start_single:] + underscores[:start_single]

            # [names] + [single underscore] + [double underscore]
            names = names[start:] + underscores
        return names

    return _sort_completions


def optimize_Completion_complete():
    from jedi.api.completion import (Completion,
                                     _extract_string_while_in_string,
                                     complete_dict,
                                     complete_file_name,
                                     filter_names)

    from .completions import _sort_completions

    def complete(self: Completion):
        leaf = self._module_node.get_leaf_for_position(
            self._original_position,
            include_prefixes=True
        )
        string, start_leaf, quote = _extract_string_while_in_string(leaf, self._original_position)

        prefixed_completions = complete_dict(
            self._module_context,
            self._code_lines,
            start_leaf or leaf,
            self._original_position,
            None if string is None else quote + string,
            fuzzy=self._fuzzy,
        )

        if string is not None:
            if not prefixed_completions:
                prefixed_completions = list(complete_file_name(
                    self._inference_state, self._module_context, start_leaf, quote, string,
                    self._like_name, self._signatures_callback,
                    self._code_lines, self._original_position,
                    self._fuzzy
                ))
            if not prefixed_completions and '\n' in string:
                # Complete only multi line strings
                prefixed_completions = self._complete_in_string(start_leaf, string)
            return prefixed_completions

        cached_name, completion_names = self._complete_python(leaf)

        completions = filter_names(self._inference_state, completion_names,
            self.stack, self._like_name, self._fuzzy, cached_name=cached_name)

        return prefixed_completions + _sort_completions(completions)

    _patch_function(Completion.complete, complete)


def optimize_filter_names():
    from jedi.api.completion import filter_names
    from jedi.api.classes import Completion
    from textension.utils import _patch_function, _named_index, _forwarder, Aggregation
    from collections import defaultdict
    from itertools import compress, repeat
    from operator import itemgetter, attrgetter, methodcaller, eq
    from builtins import map, list, zip

    from ..tools import state
    from jedi import settings

    class AggregateCompletionBase(Aggregation, Completion):
        _inference_state = state

        string_name            = _named_index(0)
        _same_name_completions = _named_index(1)
        _string_name_lower     = _named_index(2)

        __getitem__ = _forwarder("_same_name_completions.__getitem__")
        _name = property(methodcaller("__getitem__", 0))

    lower  = str.lower
    strlen = str.__len__
    get_string = attrgetter("string_name")

    def filter_names_o(inference_state, completions, stack, like_name, is_fuzzy, cached_name):
        like_name_len = strlen(like_name)

        class AggregateCompletion(AggregateCompletionBase):
            _is_fuzzy = is_fuzzy
            _like_name_length = like_name_len
            _cached_name = cached_name
            _stack = stack

        strings = map(get_string, completions)
        names = defaultdict(list)

        # Trailer completion. Show all names.
        if like_name:
            if settings.case_insensitive_completion:
                strings = map(lower, strings)
                like_name = lower(like_name)

            strings = map(itemgetter(slice(None, like_name_len)), strings)
            strings = map(eq, repeat(like_name), strings)

        for name in compress(completions, strings):
            names[name.string_name] += name,

        return map(AggregateCompletion, zip(names, names.values(), map(lower, names)))

    _patch_function(filter_names, filter_names_o)
