from src.data.utils.alphabet import Alphabet
from src.utils.constants import *

def generate_alphabet(
    dataset_type,
    other_charset = None,
    invert_charset_order=False,
):
    if other_charset is not None:
        if other_charset in CHARS.keys():
            extra_chars = CHARS[other_charset]
        else:
            extra_chars = other_charset
        all_chars = CHARS[dataset_type.lower()] if not invert_charset_order else extra_chars
        for char in extra_chars if not invert_charset_order else CHARS[dataset_type.lower()]:
            if char not in all_chars:
                all_chars += char
        alphabet = Alphabet(characters=all_chars, mode="both")
    else:
        alphabet = Alphabet(characters=CHARS[dataset_type.lower()], mode="both")

    return alphabet