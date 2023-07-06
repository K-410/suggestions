from textension.utils import _patch_function, factory

def apply():
    optimize_filter_names()
    optimize_Completion_complete()


@factory
def _sort_completions(completions):
    from itertools import compress, count, repeat
    from builtins import map, next, sorted
    from operator import attrgetter, not_

    get_string = attrgetter("_name.string_name")
    startswith = str.startswith
    rep_single = repeat("_")
    rep_double = repeat("__")

    def _sort_completions(completions):
        completions = sorted(completions, key=get_string)

        it = map(startswith, map(get_string, completions), rep_single)
        head = count()

        if i_u := next(compress(head, it), None):
            upper = completions[:i_u]
        else:
            it = map(startswith, map(get_string, completions), rep_single)
            head = count()
            upper = []

        if i_s := next(compress(head, map(not_, it)), None):
            small = completions[i_s:]
            underscores = completions[i_u:i_s]

            it = map(startswith, map(get_string, underscores), rep_double)

            if i_d := next(compress(count(), map(not_, it)), None):
                underscores = underscores[i_d:] + underscores[:i_d]
            completions = upper + small + underscores
        return completions

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

        # XXX: Natural sorting. Might want to revert to this.
        # lower = str.lower
        # def sorter(x):
        #     name = lower(x._name.string_name)
        #     return name[:2] == "__", name[0] is "_", name
        # completions = sorted(completions, key=sorter)  # 3.6ms

        return prefixed_completions + _sort_completions(completions)

    _patch_function(Completion.complete, complete)


def optimize_filter_names():
    from jedi.api.completion import filter_names
    from jedi.api.classes import Completion
    from jedi.api.helpers import _fuzzy_match
    from jedi import settings
    from textension.utils import noop_noargs, _patch_function, _named_index, consume
    from ..tools import state

    class NewCompletion(Completion):
        _is_fuzzy = False
        _inference_state = state

    class FuzzyCompletion(Completion):
        _is_fuzzy = True
        complete = property(noop_noargs)
        _inference_state = state

    fuzzy_match = _fuzzy_match
    startswith = str.startswith
    lower = str.lower
    strip = str.__str__

    from operator import itemgetter, attrgetter, eq
    from itertools import compress, repeat
    from builtins import map, list, dict, zip

    strlen = str.__len__
    get_string_name = attrgetter("string_name")
    get_tree_name = attrgetter("tree_name")

    def filter_names_o(inference_state, completions, stack, like_name, fuzzy, cached_name):
        like_name_len = strlen(like_name)

        if fuzzy:
            match_func = fuzzy_match
            completion_base = FuzzyCompletion
            do_complete = None.__init__         # Dummy
        else:
            match_func = startswith
            completion_base = NewCompletion
            # if settings.add_bracket_after_function:
            #     def do_complete():
            #         if new.type == "function":
            #             return n[like_name_len:] + "("
            #         return n[like_name_len:]
            # else:
            #     def do_complete():
            #         return n[like_name_len:]

        class completion(tuple, completion_base):
            __init__ = object.__init__
            _like_name_length = like_name_len   # Static
            _cached_name = cached_name          # Static
            _stack = stack                      # Static
            _name = _named_index(0)

        if settings.case_insensitive_completion:
            case = lower
            like_name = case(like_name)
        else:
            case = strip  # Dummy

        ret = []
        strings = map(get_string_name, completions)


        if not like_name:
            ret = list(dict(zip(strings, completions)).values())

        else:
            strings = map(itemgetter(slice(None, like_name_len)), strings)
            if settings.case_insensitive_completion:
                strings = map(lower, strings)
            strings = map(eq, repeat(like_name), strings)

            dct = {}
            for name in compress(completions, strings):
                string_name = name.string_name
                if string_name not in dct:
                    dct[string_name] = None
                    ret += [name]

            removals = []
            for name in filter(get_tree_name, ret):
                if name.tree_name.parent.type == "del_stmt":
                    removals += [name]
            if removals:
                consume(map(ret.remove, removals))

        return map(completion, zip(ret))

    _patch_function(filter_names, filter_names_o)
