
from textension.utils import _patch_function, inline
from functools import partial


def apply():
    optimize_tokenize_compile()
    optimize_tokenize_lines()
    optimize_tokenize_all_string_prefixes()


# Optimizes ``tokenize._compile`` to pass the underlying RegexFlag.UNICODE
# value to avoid dynamic class attribute access overhead.
def optimize_tokenize_compile():
    from parso.python import tokenize
    import re

    tokenize._compile = partial(re._compile, flags=re.UNICODE.value)


def optimize_tokenize_all_string_prefixes():
    from parso.python.tokenize import _all_string_prefixes
    from textension.utils import starchain
    from itertools import product, permutations
    from functools import reduce

    @inline
    def map_join(cases):
        return partial(map, "".join)
    
    @inline
    def map_upper(s):
        return partial(map, str.upper)

    def different_case_versions(prefix):
        return map_join(reduce(product, zip(prefix, map_upper(prefix))))

    empty_set = set()

    def all_string_prefixes(*, include_fstring=False, only_fstring=False):

        result = {""}
        prefixes = ("b", "r", "u", "br")

        if include_fstring:
            if only_fstring:
                prefixes = ("f", "fr")
                result = set()
            else:
                prefixes = ("b", "r", "u", "br", "f", "fr")
        elif only_fstring:
            return empty_set

        prefixes = starchain(map(permutations, prefixes))
        result.update(starchain(map(different_case_versions, prefixes)))
        return result

    _patch_function(_all_string_prefixes, all_string_prefixes)


def optimize_tokenize_lines() -> None:  # type: ignore
    from parso.python.tokenize import FStringNode, _close_fstring_if_necessary, \
        split_lines, _get_token_collection, tokenize_lines as _tokenize_lines, _find_fstring_string
    from textension.utils import inline, Aggregation, _named_index
    from .interpreter import PythonTokenTypes
    from itertools import chain, count, repeat
    from functools import partial
    from operator import attrgetter
    import re

    STRING         = PythonTokenTypes.STRING
    NAME           = PythonTokenTypes.NAME
    NUMBER         = PythonTokenTypes.NUMBER
    OP             = PythonTokenTypes.OP
    NEWLINE        = PythonTokenTypes.NEWLINE
    INDENT         = PythonTokenTypes.INDENT
    DEDENT         = PythonTokenTypes.DEDENT
    ENDMARKER      = PythonTokenTypes.ENDMARKER
    ERRORTOKEN     = PythonTokenTypes.ERRORTOKEN
    ERROR_DEDENT   = PythonTokenTypes.ERROR_DEDENT
    FSTRING_START  = PythonTokenTypes.FSTRING_START
    FSTRING_STRING = PythonTokenTypes.FSTRING_STRING
    FSTRING_END    = PythonTokenTypes.FSTRING_END

    class PythonToken2(Aggregation):
        type: PythonTokenTypes      = _named_index(0)
        string: str                 = _named_index(1)
        start_pos: tuple[int, int]  = _named_index(2)
        prefix: str                 = _named_index(3)

        @property
        def end_pos(self) -> tuple[int, int]:
            lines = split_lines(self.string)
            if len(lines) > 1:
                return self.start_pos[0] + len(lines) - 1, 0
            else:
                return self.start_pos[0], self.start_pos[1] + len(self.string)

    @inline
    def get_match(*args) -> re.Match: return re.Pattern.match
    @inline
    def get_end(*args) -> int: return re.Match.end
    @inline
    def get_span(*args) -> tuple[int, int]: return re.Match.span
    @inline
    def isidentifier(string: str) -> bool: return str.isidentifier
    @inline
    def strlen(string: str) -> int: return str.__len__
    @inline
    def is_multiline(quote: str) -> bool: return {'"""', "'''"}.__contains__
    @inline
    def make_tokens(sequence): return partial(map, PythonToken2)

    import keyword

    known_base = {*keyword.kwlist, *keyword.softkwlist, *__builtins__}

    operators = {*"** **= >> >>= << <<= // //= -> + += - -= * *= / /= % %= "
                  "& &= @ @= | |= ^ ^= ! != = == < <= > >= ~ : . , := ( ) "
                  "[ ] { }".split(" ")}

    non_letters = {*"( ) [ ] { } \n \r # 0 1 2 3 4 5 6 7 8 9 \" \' * > < = "
                    "/ - + % & ~ : . , ! | ^ \ @".split(" "), " "}

    triples = {*"f''' F''' fr''' fR''' Fr''' FR''' rf''' rF''' Rf''' RF''' "
                'f""" F""" fr""" fR""" Fr""" FR""" rf""" rF""" Rf""" RF"""'
                "".split(" ")}
    rep_zeros = repeat(0)
    get_quotes = attrgetter("quote")

    @inline
    def maplen(lines):
        return partial(map, strlen)

    def tokenize_lines(lines, *, version_info, indents=None, start_pos=(1, 0), is_first_token=True):

        if indents is None:
            indents = [0]

        def dedent_if_necessary(start):
            nonlocal result, indents
            yield from make_tokens(result)
            result = []

            while start < indents[-1]:
                if start > indents[-2]:
                    yield PythonToken2((ERROR_DEDENT, "", (lnum, start), ""))
                    indents[-1] = start
                    break
                del indents[-1]
                yield PythonToken2((DEDENT, "", spos, ""))

        known = known_base.copy()
        add_known = known.add

        pseudo_token, single_quoted, triple_quoted, endpats, whitespace, \
            fstring_pattern_map, always_break_tokens = _get_token_collection(version_info)

        match_main = partial(get_match, pseudo_token)
        match_whitespace = partial(get_match, whitespace)

        all_triples = triples | triple_quoted

        paren_level = 0  # count parentheses

        line     = ""
        token    = ""
        prefix   = ""
        contstr  = ""
        contline = ""
        additional_prefix = ""

        endprog  = None
        endmatch = None

        new_line = True
        contstr_start = 0
        lnum = start_pos[0] - 1

        result = []
        fstack = []
        pos  = 0
        end  = 0
        lnum = 0

        positions = rep_zeros
        if is_first_token:
            if "\ufeff" in lines[0][:1]:
                lines[0] = lines[0][1:]
                additional_prefix = "\ufeff"

            if start_pos[1] != 0:
                positions = chain((start_pos[1],), rep_zeros)
                lines[0] = "^" * start_pos[1] + lines[0]

        for lnum, pos, end, line in zip(count(start_pos[0]), positions, maplen(lines), lines):
            if contstr:
                if endmatch := get_match(endprog, line):
                    pos = get_end(endmatch)
                    result += (STRING, contstr + line[:pos], contstr_start, prefix),
                    contstr  = ""
                    contline = ""
                else:
                    contstr  += line
                    contline += line
                    continue

            while pos < end:
                if fstack:
                    tos = fstack[-1]
                    if not tos.is_in_expr():
                        string, pos = _find_fstring_string(endpats, fstack, line, lnum, pos)
                        if string:
                            result += (FSTRING_STRING, string, tos.last_string_start_pos, ""),
                            tos.previous_lines = ""
                            continue
                        if pos == end:
                            break

                    fstring_end_token, additional_prefix, quote_length = _close_fstring_if_necessary(
                        fstack, line[pos:], lnum, pos, additional_prefix)
                    pos += quote_length
                    if fstring_end_token:
                        result += fstring_end_token,
                        continue

                    string_line = line
                    for fstring_stack_node in fstack:
                        quote = fstring_stack_node.quote
                        if end_match := get_match(endpats[quote], line, pos):
                            end_match_string = end_match[0]
                            if strlen(end_match_string) - strlen(quote) + pos < strlen(string_line):
                                string_line = line[:pos] + end_match_string[:-strlen(quote)]
                    pseudomatch = match_main(string_line, pos)
                else:
                    c = line[pos]
                    if c in non_letters:  # Short circuit comparison.
                        if c is "\n":
                            if fstack and False in map(is_multiline, map(get_quotes, fstack)):
                                fstack = []
                            if not new_line and paren_level is 0 and not fstack:
                                result += (NEWLINE, "\n", (lnum, pos), prefix),
                            else:
                                additional_prefix += "\n"
                            new_line = True
                            break

                        elif c in {".", "=", ","}:
                            try:
                                d = c is not line[pos + 1]
                            except:
                                d = True

                            if d:
                                result += (OP, c, (lnum, pos), additional_prefix),
                                additional_prefix = ""
                                pos += 1
                                continue

                    pseudomatch = match_main(line, pos)

                if pseudomatch:
                    last_pos = pos
                    start, pos = get_span(pseudomatch, 2)
                    token = line[start:pos]

                    prefix = additional_prefix
                    if start and line[start - 1] in {" ", "\f", "\t"}:
                        prefix += line[last_pos:start]

                    additional_prefix = ""

                    if token is "":
                        additional_prefix = prefix
                        break

                    initial = token[0]
                else:
                    match = match_whitespace(line, pos)
                    start = pos = get_end(match)
                    initial = line[start]

                spos = (lnum, start)

                if new_line and initial not in {"\r", "\n", "#", "\\"}:
                    new_line = False
                    if start != indents[-1] and paren_level is 0 and not fstack:
                        if start > indents[-1]:
                            # Only yield DEDENT straight away. The diff
                            # parser doesn't care about indent sync.
                            result += (INDENT, "", spos, ""),
                            indents += start,
                        elif start < indents[-1]:
                            yield from dedent_if_necessary(start)

                if not pseudomatch:
                    if start != indents[-1] and paren_level is 0 and not fstack and start < indents[-1]:
                        yield from dedent_if_necessary(start)
                    new_line = False
                    result += (ERRORTOKEN, line[pos], spos, additional_prefix + match[0]),  # type: ignore
                    additional_prefix = ""
                    pos += 1
                    continue

                elif initial not in non_letters and (token in known or (isidentifier(token) and not add_known(token))):
                    if token in always_break_tokens and (fstack or paren_level):
                        fstack = []
                        paren_level = 0
                        if m := re.match(r'[ \f\t]*$', line[:start]):
                            start = get_end(m)
                            if start < indents[-1]:
                                yield from dedent_if_necessary(start)
                    result += (NAME, token, spos, prefix),

                elif initial in {"\n", "\r"}:
                    if fstack and False in map(is_multiline, map(get_quotes, fstack)):
                        fstack = []
                    if not new_line and paren_level is 0 and not fstack:
                        result += (NEWLINE, token, spos, prefix),
                    else:
                        additional_prefix = prefix + token
                    new_line = True

                elif token in operators:
                    if fstack and token[0] is ":" and fstack[-1].parentheses_count - fstack[-1].format_spec_count is 1:
                        fstack[-1].format_spec_count += 1
                        token = ':'
                        pos = start + 1
                    elif token in {"(", ")", "[", "]", "{", "}"}:
                        if token in {"(", "[", "{"}:
                            if fstack:
                                fstack[-1].open_parentheses(token)
                            else:
                                paren_level += 1
                        else:
                            if fstack:
                                fstack[-1].close_parentheses(token)
                            else:
                                if paren_level:
                                    paren_level -= 1
                    result += (OP, token, spos, prefix),

                elif initial is "#":
                    if fstack and fstack[-1].is_in_expr():
                        result += (ERRORTOKEN, initial, spos, prefix),
                        pos = start + 1
                    else:
                        additional_prefix = prefix + token

                elif token in fstring_pattern_map:  # The start of an fstring.
                    fstack += FStringNode(fstring_pattern_map[token]),
                    result += (FSTRING_START, token, spos, prefix),

                # Unprefixed strings are most common and checked first.
                elif initial in {"\"", "\'"}:
                    if token in triple_quoted:
                        endprog = endpats[token]
                        if endmatch := get_match(endprog, line, pos):  # all on one line
                            pos = get_end(endmatch)
                            token = line[start:pos]
                            result += (STRING, token, spos, prefix),
                        else:
                            contstr_start = spos                    # multiple lines
                            contstr = line[start:]
                            contline = line
                            break
                    elif token[-1] is "\n":
                        contstr_start = spos
                        endprog = endpats.get(initial) or endpats.get(token[1]) or endpats.get(token[2])
                        contstr = line[start:]
                        contline = line
                        break
                    else:                                       # ordinary string
                        result += (STRING, token, spos, prefix),

                elif (initial in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"} or \
                   (initial is "." != token != "...")):
                    result += (NUMBER, token, spos, prefix),

                elif initial is '\\' and line[start:] in {"\\\n", "\\\r\n", "\\\r"}:  # continued stmt
                    additional_prefix += prefix + line[start:]
                    break

                elif token in all_triples:
                    if token in triple_quoted:
                        endprog = endpats[token]
                        if endmatch := get_match(endprog, line, pos):
                            pos = get_end(endmatch)
                            token = line[start:pos]
                            result += (STRING, token, spos, prefix),
                            continue
                        contstr_start = spos                    # multiple lines
                        contstr = line[start:]
                        contline = line
                        break
                    fstack += FStringNode(fstring_pattern_map[token]),
                    result += (FSTRING_START, token, spos, prefix),

                else:
                    if token not in  {"...", ";"}:
                        if   initial in single_quoted or \
                           token[:2] in single_quoted or \
                           token[:3] in single_quoted:
                            if token[-1] in {"\r", "\n"}:                       # continued string
                                # This means that a single quoted string ends with a
                                # backslash and is continued.
                                contstr_start = lnum, start
                                endprog = (endpats.get(initial) or endpats.get(token[1])
                                        or endpats.get(token[2]))
                                contstr = line[start:]
                                contline = line
                                break
                            else:                                       # ordinary string
                                result += (STRING, token, spos, prefix),
                                continue
                        else:
                            print("unhandled OP token:", repr(token), spos, prefix)
                    result += (OP, token, spos, prefix),

            if result:
                yield from make_tokens(result)
                result = []

        if contstr:
            result += (ERRORTOKEN, contstr, contstr_start, prefix),

        if fstack:
            tos = fstack[-1]
            if tos.previous_lines:
                result += (FSTRING_STRING, tos.previous_lines, tos.last_string_start_pos, ""),

        yield from make_tokens(result)

        end_pos = lnum, end

        for _ in indents[1:]:
            del indents[-1]
            yield PythonToken2((DEDENT, "", end_pos, ""))
        yield PythonToken2((ENDMARKER, "", end_pos, additional_prefix))

        return None

    _patch_function(_tokenize_lines, tokenize_lines)

