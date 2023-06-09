from textension.utils import _patch_function

def apply():
    optimize_filter_names()
    optimize_Completion_complete()


def optimize_Completion_complete():
    from jedi.api.completion import (Completion,
                                     _extract_string_while_in_string,
                                     complete_dict,
                                     complete_file_name,
                                     filter_names,
                                     _remove_duplicates)

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

        if string is not None and not prefixed_completions:
            prefixed_completions = list(complete_file_name(
                self._inference_state, self._module_context, start_leaf, quote, string,
                self._like_name, self._signatures_callback,
                self._code_lines, self._original_position,
                self._fuzzy
            ))
        if string is not None:
            if not prefixed_completions and '\n' in string:
                # Complete only multi line strings
                prefixed_completions = self._complete_in_string(start_leaf, string)
            return prefixed_completions

        cached_name, completion_names = self._complete_python(leaf)

        completions = filter_names(self._inference_state, completion_names,
            self.stack, self._like_name, self._fuzzy, cached_name=cached_name)

        lower = str.lower
        def sorter(x):
            name = lower(x.name)
            return name[:2] == "__", name[0] is "_", name

        if prefixed_completions:
            _remove_duplicates(prefixed_completions, completions)

        # Removing duplicates mostly to remove False/True/None duplicates.
        # return prefixed_completions + sorted(completions, key=sorter)
        return prefixed_completions + sorted(completions, key=sorter)

    _patch_function(Completion.complete, complete)


def optimize_filter_names():
    from jedi.api.completion import filter_names
    from jedi.api.classes import Completion
    from jedi.api.helpers import _fuzzy_match
    from jedi import settings
    from textension.utils import noop_noargs, _patch_function, _named_index
    from ..tools import state

    class NewCompletion(Completion):
        _is_fuzzy = False
        _inference_state = state

    class FuzzyCompletion(Completion):
        _is_fuzzy = True
        complete = property(noop_noargs)
        _inference_state = state

    _len = len
    fuzzy_match = _fuzzy_match
    startswith = str.startswith
    lower = str.lower
    strip = str.__str__

    from operator import attrgetter
    get_name = attrgetter("string_name")

    def filter_names_o(inference_state, completion_names, stack, like_name, fuzzy, cached_name):
        like_name_len = _len(like_name)
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
        dct = {}

        if not like_name:
            d = dict(zip(map(get_name, completion_names), completion_names))
            for e in d.values():
                if tn := e.tree_name:
                    if tn.parent.type == "del_stmt":
                        continue
                ret += [(e,)]

        else:
            for e in completion_names:
                n_ = e.string_name
                if case(n_[0]) == like_name[0]:
                    if case(n_[:like_name_len]) == like_name:
                        # Store unmodified so names like Cache and cache aren't merged.
                        if (k := (n_, n_[like_name_len:])) not in dct:
                            dct[k] = None

                            if tn := e.tree_name:
                                if tn.parent.type == "del_stmt":
                                    continue
                            ret += [(e,)]
        return map(completion, ret)

    _patch_function(filter_names, filter_names_o)
