import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.utils.subsequent_mask import subsequent_mask
from src.utils.constants import *


def collate(batch):
    imgs_in = []
    imgs_plain = []
    imgs_len = []
    texts_in = []
    text_logits_ctc_in = []
    text_logits_s2s_in = []
    texts_len = []
    writers = []
    names = list()
    paragraph_ids = list()
    for item in batch:
        names.append(item[NAME])
        paragraph_ids.append(item[PARAGRAPH_ID]) if PARAGRAPH_ID in item.keys() else paragraph_ids.append(None)
        imgs_in.append(item[LINE_IMAGE].permute(2,0,1))
        if PLAIN_IMAGE in item.keys():
            imgs_plain.append(item[PLAIN_IMAGE].convert("RGB"))
        if UNPADDED_IMAGE_WIDTH in item.keys():
            imgs_len.append(item[UNPADDED_IMAGE_WIDTH])
        else:
            imgs_len.append(item[LINE_IMAGE].shape[-1])
        texts_in.append(item[TEXT])
        text_logits_ctc_in.append(item[TEXT_LOGITS_CTC])
        texts_len.append(len(item[TEXT]))
        if TEXT_LOGITS_S2S in item.keys():
            text_logits_s2s_in.append(item[TEXT_LOGITS_S2S])
        writers.append(item[WRITER])
    imgs_out = pad_sequence(imgs_in).permute(1,2,3,0)
    text_logits_ctc_out = pad_sequence(text_logits_ctc_in, padding_value=1, batch_first=True)
    out_dict = {
        NAME: names,
        PARAGRAPH_ID: paragraph_ids,
        LINE_IMAGE: imgs_out,
        TEXT: texts_in,
        TEXT_LOGITS_CTC: text_logits_ctc_out,
        WRITER: torch.LongTensor(writers),
        UNPADDED_IMAGE_WIDTH: torch.LongTensor(imgs_len),
        UNPADDED_TEXT_LEN: torch.LongTensor(texts_len),
        SRC_KEY_PADDING_MASK: torch.Tensor([i.shape[0] for i in imgs_in])/imgs_out.shape[-1],
    }
    if len(text_logits_s2s_in)>0:
        text_logits_s2s_out = pad_sequence(text_logits_s2s_in, padding_value=1, batch_first=True)
        out_dict[TEXT_LOGITS_S2S] = text_logits_s2s_out
        out_dict[TGT_KEY_PADDING_MASK] = torch.eq(text_logits_s2s_out,
                                                  torch.ones(text_logits_s2s_out.shape, dtype=torch.long) * torch.LongTensor([1]))
        out_dict[TGT_MASK] = subsequent_mask(text_logits_s2s_out.shape[-1] - 1)
    if TEXT_DIPLOMATIC in batch[0].keys():
        texts_dipl_in = []
        texts_dipl_logits_ctc_in = []
        texts_dipl_logits_s2s_in = []
        for item in batch:
            texts_dipl_in.append(item[TEXT_DIPLOMATIC])
            texts_dipl_logits_ctc_in.append(item[TEXT_DIPLOMATIC_LOGITS_CTC])
            if TEXT_DIPLOMATIC_LOGITS_S2S in item.keys():
                texts_dipl_logits_s2s_in.append(item[TEXT_DIPLOMATIC_LOGITS_S2S])
        texts_dipl_logits_ctc_out = pad_sequence(texts_dipl_logits_ctc_in, padding_value=1, batch_first=True)
        out_dict[TEXT_DIPLOMATIC] = texts_dipl_in
        out_dict[TEXT_DIPLOMATIC_LOGITS_CTC] = texts_dipl_logits_ctc_out
        if len(texts_dipl_logits_s2s_in)>0:
            texts_dipl_logits_s2s_out = pad_sequence(texts_dipl_logits_s2s_in, padding_value=1, batch_first=True)
            out_dict[TEXT_DIPLOMATIC_LOGITS_S2S] = texts_dipl_logits_s2s_out
            out_dict[TGT_KEY_PADDING_MASK_DIPL] = torch.eq(texts_dipl_logits_s2s_out,
                                                        torch.ones(texts_dipl_logits_s2s_out.shape,
                                                                     dtype=torch.long) * torch.LongTensor([1]))
            out_dict[TGT_MASK_DIPL] = subsequent_mask(texts_dipl_logits_s2s_out.shape[-1] - 1)

    if PLAIN_IMAGE in batch[0].keys():
        out_dict[PLAIN_IMAGE] = imgs_plain
    if PRIMING_TEXT in batch[0].keys():
        p_imgs_in = []
        p_imgs_len = []
        p_texts_in = []
        p_text_logits_ctc_in = []
        p_text_logits_s2s_in = []
        p_texts_len = []
        for item in batch:
            p_imgs_in.append(item[PRIMING_LINE_IMAGE].permute(2, 0, 1))
            if PRIMING_UNPADDED_IMAGE_WIDTH in item.keys():
                p_imgs_len.append(item[PRIMING_UNPADDED_IMAGE_WIDTH])
            else:
                p_imgs_len.append(item[PRIMING_LINE_IMAGE].shape[-1])
            p_texts_in.append(item[PRIMING_TEXT])
            p_text_logits_ctc_in.append(item[PRIMING_TEXT_LOGITS_CTC])
            p_texts_len.append(len(item[PRIMING_TEXT]))
            if PRIMING_TEXT_LOGITS_S2S in item.keys():
                p_text_logits_s2s_in.append(item[PRIMING_TEXT_LOGITS_S2S])
            p_imgs_out = pad_sequence(p_imgs_in).permute(1, 2, 3, 0)
            p_text_logits_ctc_out = pad_sequence(p_text_logits_ctc_in, padding_value=1, batch_first=True)
        out_dict[PRIMING_LINE_IMAGE] = p_imgs_out
        out_dict[PRIMING_TEXT] = p_texts_in
        out_dict[PRIMING_TEXT_LOGITS_CTC] = p_text_logits_ctc_out
        out_dict[PRIMING_UNPADDED_IMAGE_WIDTH] = torch.LongTensor(p_imgs_len)
        out_dict[PRIMING_UNPADDED_TEXT_LEN] = torch.LongTensor(p_texts_len)
        out_dict[PRIMING_SRC_KEY_PADDING_MASK] = torch.Tensor([i.shape[0] for i in p_imgs_in])/p_imgs_out.shape[-1]
        if len(p_text_logits_s2s_in)>0:
            p_text_logits_s2s_out = pad_sequence(p_text_logits_s2s_in, padding_value=1, batch_first=True)
            out_dict[PRIMING_TEXT_LOGITS_S2S] = p_text_logits_s2s_out
            out_dict[PRIMING_TGT_KEY_PADDING_MASK] = torch.eq(p_text_logits_s2s_out,
                                                      torch.ones(p_text_logits_s2s_out.shape,
                                                                 dtype=torch.long) * torch.LongTensor([1]))
    return out_dict


def slim_collate(batch):
    imgs = []
    texts = []
    text_logits_ctc_in = []
    text_logits_s2s_in = []
    texts_len = []
    writers = []
    names = []
    for item in batch:
        imgs.append(item[LINE_IMAGE])
        texts.append(item[TEXT])
        text_logits_ctc_in.append(item[TEXT_LOGITS_CTC])
        texts_len.append(len(item[TEXT]))
        writers.append(item[WRITER])
        names.append(item[NAME])
    text_logits_ctc_out = pad_sequence(text_logits_ctc_in, padding_value=1, batch_first=True)
    out_dict = {
        NAME: names,
        LINE_IMAGE: imgs,
        TEXT: texts,
        TEXT_LOGITS_CTC: text_logits_ctc_out,
        WRITER: torch.LongTensor(writers),
        UNPADDED_TEXT_LEN: torch.LongTensor(texts_len),
    }
    if len(text_logits_s2s_in) > 0:
        text_logits_s2s_out = pad_sequence(text_logits_s2s_in, padding_value=1, batch_first=True)
        out_dict[TEXT_LOGITS_S2S] = text_logits_s2s_out
        out_dict[TGT_KEY_PADDING_MASK] = torch.eq(text_logits_s2s_out,
                                                  torch.ones(text_logits_s2s_out.shape,
                                                             dtype=torch.long) * torch.LongTensor([1]))
        out_dict[TGT_MASK] = subsequent_mask(text_logits_s2s_out.shape[-1] - 1)
    return out_dict
