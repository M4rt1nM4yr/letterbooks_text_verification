import torch
from src.utils.constants import *

def ctc_remove_successives_identical_ind(ind, remove_ind=None):
    res = []
    for i in ind:
        if res and res[-1]==i:
            continue
        res.append(i)
    out = []
    for r in res:
        if r in remove_ind:
            continue
        out.append(r)
    return torch.LongTensor(out)

def ctc_remove_successives_from_batch(ind, remove_ind=None, widths=None):
    out = []
    for idx,i in enumerate(ind):
        if widths is not None:
            i = i[:widths[idx]]
        out.append(ctc_remove_successives_identical_ind(i, remove_ind=remove_ind))
    return out

def ctc_pred_to_text(pred, widths, alphabet, batch_first=True):
    if batch_first is False:
        pred = pred.permute(1,0,2)
    if isinstance(pred, list):
        pred_str = list()
        for p, w in zip(pred, widths):
            pred_logits = torch.argmax(p, dim=1).long().detach()
            pred_logits_clean = ctc_remove_successives_from_batch(pred_logits.unsqueeze(0), [alphabet.toPosition[PAD], alphabet.toPosition[BLANK]], widths=w.unsqueeze(0))
            out, _ = alphabet.batch_logits_to_string_list(pred_logits_clean)
            pred_str.append(out[0])
    else:
        pred_logits = torch.argmax(pred, dim=2).long().detach()
        pred_logits_clean = ctc_remove_successives_from_batch(pred_logits, [alphabet.toPosition[PAD], alphabet.toPosition[BLANK]], widths=widths)
        pred_str, _ = alphabet.batch_logits_to_string_list(pred_logits_clean)
    return pred_str


if __name__ == "__main__":
    pred = torch.randn(2,25,5)
    pred_logits = torch.argmax(pred, dim=2).long()
    widths = torch.LongTensor([10,14])
    print(pred_logits)
    pred_logits_cleaned = ctc_remove_successives_from_batch(pred_logits, remove_ind=[0,1], widths=widths)
    print(pred_logits_cleaned, len(pred_logits_cleaned[0]))

    from src.data.utils.alphabet import Alphabet
    chars = string.ascii_lowercase
    alphabet = Alphabet(chars)
    print(ctc_pred_to_text(pred.permute(1,0,2), widths, alphabet, batch_first=False))