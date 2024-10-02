import re

from src.utils.constants import *

def clean_string_basic(str_in):
    text = str_in.replace(u'\u0056\u0308', u'\ue342')
    text = text.replace(u'\u0076\u0308', u'\ue742')
    text = text.replace('å', 'ä')
    text = text.replace(u'\u0065\u0308', '\u00eb')
    text = text.replace(u'\u0045\u0308', '\u00cb')
    text = text.replace("ẅ", "w")
    text = text.replace(u'\u0079\u0308','\u00ff')
    text = text.replace(u'\u0059\u0308','\u0178')
    text = text.replace('ÿ',"y")
    text = text.replace('<del rend="strikethrough">', DEL_OPEN)
    text = text.replace('<del rend="erased">', DEL_ERASED_OPEN)
    text = text.replace('<del>', DEL_OPEN)
    text = text.replace('</del>', DEL_CLOSE)
    text = text.strip()
    assert "&" not in text
    assert "$" not in text
    return text