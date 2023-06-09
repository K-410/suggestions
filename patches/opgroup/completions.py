def apply():
    optimize_filter_names()


def optimize_filter_names():
    from jedi.api.completion import filter_names
    from jedi.api.classes import Completion
    from jedi.api.helpers import _fuzzy_match
    from jedi import settings
    from textension.utils import noop_noargs, _patch_function
    from ..tools import state

    class NewCompletion(Completion):
        __init__ = noop_noargs
        _is_fuzzy = False
        _inference_state = state

    class FuzzyCompletion(Completion):
        __init__ = noop_noargs
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

        class completion(completion_base):
            _like_name_length = like_name_len   # Static
            _cached_name = cached_name          # Static
            _stack = stack                      # Static


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
                tn = e.tree_name
                if tn is None or (getattr(tn.get_definition(), "type", None) != "del_stmt"):
                    new = completion()
                    new._name = e
                    ret += [new]

            # for e in completion_names:
            #     n_ = e.string_name
            #     if n_ not in dct:
            #         dct[n_] = None
            #         if ((tn := e.tree_name) is None) or (getattr(tn.get_definition(), "type", None) != "del_stmt"):
            #             new = completion()
            #             new._name = e
            #             ret += [new]

        else:
            for e in completion_names:
                n_ = e.string_name
                if case(n_[0]) == like_name[0]:
                    if case(n_[:like_name_len]) == like_name:
                        # Store unmodified so names like Cache and cache aren't merged.
                        if (k := (n_, n_[like_name_len:])) not in dct:
                            dct[k] = None
                            if ((tn := e.tree_name) is None) or (getattr(tn.get_definition(), "type", None) != "del_stmt"):
                                new = completion()
                                new._name = e
                                ret += [new]
        return ret

    _patch_function(filter_names, filter_names_o)
