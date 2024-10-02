import  string

BLANK = "Δ"
PAD = "Φ"
START_OF_SEQUENCE = "Α"
END_OF_SEQUENCE = "Ω"

EXPAN_OPEN = "Ψ"
EXPAN_CLOSE = "ψ"
EX_OPEN = "Τ"
EX_CLOSE = "τ"
HI_OPEN = "Λ"
HI_CLOSE = "λ"
DEL_OPEN = "Ξ"
DEL_CLOSE = "ξ"
DEL_CROSSOUT_OPEN = "Χ"
DEL_ERASED_OPEN = "χ"
ADD_OPEN = "Ε"
ADD_CLOSE = "ε"
HI_UNDERLINED_OPEN = "Η"


NAME = "name"
PARAGRAPH_ID = "paragraph_id"
LINE_IMAGE = "line_image"
PLAIN_IMAGE = "plain_image"
PATCH_IMAGE = "patch_image"
NUM_PATCHES = "num_patches"
PATCH_LOCATIONS = "patch_locations"
PRIMING_PATCH_LOCATIONS = "priming_patch_locations"
WRITER = "writer"
TEXT = "text"
TEXT_LOGITS_CTC = "text_logits_ctc"
TEXT_LOGITS_S2S = "text_logits_s2s"
TEXT_DIPLOMATIC = "text_diplomatic"
TEXT_DIPLOMATIC_LOGITS_CTC = "text_diplomatic_logits_ctc"
TEXT_DIPLOMATIC_LOGITS_S2S = "text_diplomatic_logits_s2s"
TGT_KEY_PADDING_MASK_DIPL = "tgt_key_padding_mask_dipl"
TGT_MASK_DIPL = "tgt_mask_dipl"
SRC_KEY_PADDING_MASK = "src_key_padding_mask"
PRIMING_SRC_KEY_PADDING_MASK = "priming_src_key_padding_mask"
TGT_KEY_PADDING_MASK = "tgt_key_padding_mask"
TGT_MASK = "tgt_mask"
PRIMING_NAME = "priming_name"
PRIMING_LINE_IMAGE = "priming_line_image"
PRIMING_PATCH_IMAGE = "priming_patch_image"
PRIMING_TEXT = "priming_text"
PRIMING_TEXT_LOGITS_CTC = "priming_text_logits_ctc"
PRIMING_TEXT_LOGITS_S2S = "priming_text_logits_s2s"
PRIMING_TGT_KEY_PADDING_MASK = "priming_tgt_key_padding_mask"
PRIMING_UNPADDED_IMAGE_WIDTH = "priming_unpadded_image_width"
PRIMING_UNPADDED_TEXT_LEN = "priming_unpadded_text_len"
PRIMING_SRC_KEY_PADDING_MASK = "priming_src_key_padding_mask"

UNPADDED_IMAGE_WIDTH = "unpadded_image_width"
UNPADDED_TEXT_LEN = "unpadded_text_len"

nbb_extra_all = "".join(['ſ', 'ʒ', 'Ʒ', 'Ü', 'ü', 'Ö', 'ö', 'Ä', 'ä', '.',':',',','/','~','§','%','-','*',' ','@'])
nbb_extra = "".join(['ſ', 'ʒ', 'Ʒ', 'Ü', 'ü', 'Ö', 'ö', 'Ä', 'ä', '.',':',',','/','~','-','*',' '])
NBB_CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + nbb_extra
nbb_dipl_extra = "".join(['ı', 'ȷ', '%', 'ë', ';', '=', '$',
                          "\u00ff", "\u1e8f", "\u0308", '\u0303', '\u0363', '\u005e', '\u0366', '\u0302', '\u0367',
                          '\u0307', '\u0364', 'Ɉ', 'ɉ', 'ɟ', 'ꝛ',
                          '+', '`', '(',  '@', 'ẅ', "\ue742", '\ue342', '§', 'Ë',
                          '>', 'ß', '&', '?', ')', '\t',
                          EXPAN_OPEN, EXPAN_CLOSE, EX_OPEN, EX_CLOSE, HI_OPEN, HI_CLOSE, DEL_OPEN, DEL_CLOSE,
                          DEL_ERASED_OPEN, DEL_CROSSOUT_OPEN, ADD_OPEN, ADD_CLOSE, HI_UNDERLINED_OPEN])
NBB_DIPL_CHARS = NBB_CHARS + nbb_dipl_extra 

CHARS = {
    "nbb": NBB_CHARS,
    "nbb_dipl": NBB_DIPL_CHARS,
}