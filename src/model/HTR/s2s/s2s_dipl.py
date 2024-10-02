from functools import partial
from typing import Any, Optional

import einops
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics import CharErrorRate, WordErrorRate

from src.data.utils.ctc_helper import ctc_pred_to_text
from src.data.utils.extract_abbreviations import extract_abbreviations
from src.data.utils.noisy_teacher_forcing import NoisyTeacherForcing
from src.model.HTR.encoder.encoder_wrapper import EncoderWrapper
from src.model.utils.positional_encoding.positional_encoding import Summer, PositionalEncoding1D
from src.utils.constants import *

class S2SDipl(pl.LightningModule):
    def __init__(
            self,
            alphabet,
            cnn,
            encoder,
            decoder,
            scheduler=None,
            lr=1e-3,
            label_smoothing=0.,
            p_noisy_teacher_forcing=0.,
            metric="val/cer_s2s",
            decoder_warm_up_epochs=0,
            fast_val=False,
            transcription_target = "dipl",
            warm_up_steps=1024,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['cnn','encoder','decoder', 'alphabet'])
        self.alphabet = alphabet
        self.fast_val = fast_val
        self.transcription_target = transcription_target
        self.criterion_ctc = nn.CTCLoss(zero_infinity=True)
        self.criterion_s2s = nn.CrossEntropyLoss(
            ignore_index=self.alphabet.toPosition[PAD],
            label_smoothing=label_smoothing,
        )
        self.noisy_teacher_forcing = nn.Identity() if p_noisy_teacher_forcing == 0. else NoisyTeacherForcing(
            vocab_size=len(alphabet.toPosition),
            noise_prob=p_noisy_teacher_forcing,
        )
        self.decoder_warm_up_epochs = decoder_warm_up_epochs
        self.decoder_warm_up_factor = 1.

        self.cnn = cnn
        self.encoder = EncoderWrapper(
            encoder,
            in_channels=cnn.out_dim*cnn.height_at_64px,   # because of fine-grained CNN
        )
        memory_dim = self.encoder.out_dim
        self.decoder = decoder(
            alphabet=alphabet,
            memory_dim=memory_dim,
        ) if isinstance(decoder, partial) else decoder
        self.pe1d = nn.Identity()
        self.predictor_ctc = nn.Sequential(
            nn.Linear(self.encoder.out_dim, len(alphabet.toPosition)),
            nn.LogSoftmax(dim=-1)
        )
        self.cer_s2s_val = CharErrorRate()
        self.wer_s2s_val = WordErrorRate()
        self.cer_s2s_val_cheat = CharErrorRate()
        self.wer_s2s_val_cheat = WordErrorRate()
        self.cer_ctc_val = CharErrorRate()
        self.wer_ctc_val = WordErrorRate()
        self.cer_s2s_abbr_val = CharErrorRate()
        self.wer_s2s_abbr_val = WordErrorRate()

        self.cer_s2s_test = CharErrorRate()
        self.wer_s2s_test = WordErrorRate()
        self.cer_ctc_test = CharErrorRate()
        self.wer_ctc_test = WordErrorRate()
        self.cer_s2s_abbr_test = CharErrorRate()
        self.wer_s2s_abbr_test = WordErrorRate()

    def forward(
            self,
            x,
            y,
            img_width=None,
            tgt_key_padding_mask=None,
            tgt_mask=None,
    ):
        enc_out = self.encode(
            x,
            img_width,
        )
        memory = enc_out["memory"]
        memory = self.pe1d(memory)
        memory_key_padding_mask = torch.zeros((memory.shape[0], memory.shape[1]), dtype=torch.bool, device=memory.device)
        for idx, w in enumerate(enc_out["lengths"]):
            memory_key_padding_mask[idx, w:] = True
        dec_out = self.decoder(
            memory=memory,
            y=y,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        return {
            "log_predictions_ctc": enc_out["predictions"],
            "predictions_s2s": dec_out["predictions"],
            "lengths_ctc": enc_out["lengths"],
        }

    @torch.no_grad()
    def inference(
            self,
            x,
            img_width=None,
            max_char_len=None,
    ):
        if img_width==None:
            img_width = torch.ones((x.shape[0]))*x.shape[-1]

        enc_out = self.encode(
            x,
            img_width,
        )
        memory = enc_out["memory"]
        memory = self.pe1d(memory)
        memory_key_padding_mask = torch.zeros((memory.shape[0], memory.shape[1]), dtype=torch.bool,
                                              device=memory.device)
        for idx, w in enumerate(enc_out["lengths"]):
            memory_key_padding_mask[idx, w:] = True
        dec_out = self.decoder.inference(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            max_char_len=max_char_len,
        )
        return {
            "log_predictions_ctc": enc_out["predictions"],
            "predictions_s2s": dec_out["predictions"],
            "lengths_ctc": enc_out["lengths"],
        }



    def encode(
            self,
            x,
            img_width=None,
    ):
        f = self.cnn(x)
        f = einops.rearrange(f, 'b c h w -> b w (c h)')
        if img_width is not None:
            lengths = torch.ceil(img_width / torch.max(img_width) * f.shape[1]).cpu().int()
        else:
            lengths = None
        out = self.encoder(
            x=f,
            lengths=lengths,
        )
        predictions = self.predictor_ctc(out)
        return {
            "features": f,
            "memory": out,
            "predictions": predictions,
            "lengths": lengths,
        }

    def on_train_epoch_start(self) -> None:
        if self.decoder_warm_up_epochs > 0:
            self.decoder_warm_up_factor = min(1., self.current_epoch / self.decoder_warm_up_epochs)

    def training_step(self, batch, batch_idx):
        name, x, width, logits_ctc = batch[NAME], batch[LINE_IMAGE], batch[UNPADDED_IMAGE_WIDTH], \
        batch[TEXT_LOGITS_CTC]
        if self.transcription_target == "dipl":
            logits_target_s2s = batch[TEXT_DIPLOMATIC_LOGITS_S2S]
            tgt_mask, tgt_key_padding_mask = batch[TGT_MASK_DIPL], batch[TGT_KEY_PADDING_MASK_DIPL]
        elif self.transcription_target == "basic":
            logits_target_s2s = batch[TEXT_LOGITS_S2S]
            tgt_mask, tgt_key_padding_mask = batch[TGT_MASK], batch[TGT_KEY_PADDING_MASK]
        logits_len = batch[UNPADDED_TEXT_LEN]

        out = self(
            x,
            y = self.noisy_teacher_forcing(logits_target_s2s[:,:-1]),
            img_width=width,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:,:-1],
        )
        loss_s2s = self.criterion_s2s(
            einops.rearrange(out['predictions_s2s'], 'b s f -> (b s) f'),
            einops.rearrange(logits_target_s2s[:, 1:], 'b s -> (b s)'),
        )
        self.log("train/loss_s2s", loss_s2s.item(), on_step=False, on_epoch=True, prog_bar=True)

        if self.decoder_warm_up_epochs > 0:
            loss_s2s = min(loss_s2s, loss_s2s * self.decoder_warm_up_factor)
        self.log("train/loss", loss_s2s.item(), on_step=False, on_epoch=True, prog_bar=True)

        return loss_s2s

    def validation_step(self, batch, batch_idx):
        name, x, width, logits_ctc = batch[NAME], batch[LINE_IMAGE], batch[UNPADDED_IMAGE_WIDTH], \
            batch[TEXT_LOGITS_CTC]
        if self.transcription_target == "dipl":
            logits_target_s2s = batch[TEXT_DIPLOMATIC_LOGITS_S2S]
            tgt_mask, tgt_key_padding_mask = batch[TGT_MASK_DIPL], batch[TGT_KEY_PADDING_MASK_DIPL]
            text_target = batch[TEXT_DIPLOMATIC]
        elif self.transcription_target == "basic":
            logits_target_s2s = batch[TEXT_LOGITS_S2S]
            tgt_mask, tgt_key_padding_mask = batch[TGT_MASK], batch[TGT_KEY_PADDING_MASK]
            text_target = batch[TEXT]
        logits_len = batch[UNPADDED_TEXT_LEN]

        out_cheat = self(x, y = self.noisy_teacher_forcing(logits_target_s2s[:,:-1]), img_width=width, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:,:-1])
        if not self.fast_val:
            out = self.inference(x, width, max_char_len=logits_target_s2s.shape[-1]-1)
        else:
            out = out_cheat

        if not self.fast_val:
            loss_s2s = self.criterion_s2s(
                einops.rearrange(out['predictions_s2s'], 'b s f -> (b s) f'),
                einops.rearrange(logits_target_s2s[:, 1:],'b s -> (b s)'),
            )
            self.log("val/loss_s2s", loss_s2s.item(), on_step=False, on_epoch=True, prog_bar=True)
        loss_s2s_cheat = self.criterion_s2s(
            einops.rearrange(out_cheat['predictions_s2s'], 'b s f -> (b s) f'),
            einops.rearrange(logits_target_s2s[:, 1:],'b s -> (b s)'),
        )
        self.log("val/loss_s2s_cheat", loss_s2s_cheat.item(), on_step=False, on_epoch=True, prog_bar=True)

        if not self.fast_val:
            if self.decoder_warm_up_epochs > 0:
                loss_s2s = min(loss_s2s, loss_s2s * self.decoder_warm_up_factor)
            # loss = self.lambda_ctc * loss_ctc + (1 - self.lambda_ctc) * loss_s2s
            self.log("val/loss", loss_s2s.item(), on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss_s2s = 0

        if not self.fast_val:
            pred_text_s2s, _ = self.alphabet.batch_logits_to_string_list(torch.argmax(out["predictions_s2s"].cpu(), dim=-1),
                                                                              stopping_logits=[
                                                                                  self.alphabet.string_to_logits(
                                                                                      END_OF_SEQUENCE)])
            self.cer_s2s_val(pred_text_s2s, text_target)
            self.wer_s2s_val(pred_text_s2s, text_target)
        pred_text_s2s_cheat, _ = self.alphabet.batch_logits_to_string_list(torch.argmax(out_cheat["predictions_s2s"].cpu(), dim=-1),
                                                                          stopping_logits=[
                                                                              self.alphabet.string_to_logits(
                                                                                  END_OF_SEQUENCE)])
        self.cer_s2s_val_cheat(pred_text_s2s_cheat, text_target)
        self.wer_s2s_val_cheat(pred_text_s2s_cheat, text_target)

        self.val_pred_text_s2s = pred_text_s2s if not self.fast_val else pred_text_s2s_cheat
        self.val_pred_text_s2s_cheat = pred_text_s2s_cheat
        self.val_target_text = text_target

        if not self.fast_val and self.transcription_target == "dipl":
            pred_abbr = extract_abbreviations(pred_text_s2s, EX_OPEN, EX_CLOSE)
            tgt_abbr = extract_abbreviations(text_target, EX_OPEN, EX_CLOSE)
            self.cer_s2s_abbr_val(pred_abbr, tgt_abbr)
            self.wer_s2s_abbr_val(pred_abbr, tgt_abbr)

        return loss_s2s

    def on_validation_epoch_end(self) -> None:
        if not self.fast_val:
            self.log("val/cer_s2s", self.cer_s2s_val.compute())
            self.log("val/wer_s2s", self.wer_s2s_val.compute())
            self.cer_s2s_val.reset()
            self.wer_s2s_val.reset()
            self.log("val/cer_s2s_abbr", self.cer_s2s_abbr_val.compute())
            self.log("val/wer_s2s_abbr", self.wer_s2s_abbr_val.compute())
            self.cer_s2s_abbr_val.reset()
            self.wer_s2s_abbr_val.reset()
        self.log("val/cer_s2s_cheat", self.cer_s2s_val_cheat.compute())
        self.log("val/wer_s2s_cheat", self.wer_s2s_val_cheat.compute())
        self.cer_s2s_val_cheat.reset()
        self.wer_s2s_val_cheat.reset()


    def test_step(self, batch, batch_idx):
        name, x, width, logits_ctc = batch[NAME], batch[LINE_IMAGE], batch[UNPADDED_IMAGE_WIDTH], \
            batch[TEXT_LOGITS_CTC]
        if self.transcription_target == "dipl":
            logits_target_s2s = batch[TEXT_DIPLOMATIC_LOGITS_S2S]
            tgt_mask, tgt_key_padding_mask = batch[TGT_MASK_DIPL], batch[TGT_KEY_PADDING_MASK_DIPL]
            text_target = batch[TEXT_DIPLOMATIC]
        elif self.transcription_target == "basic":
            logits_target_s2s = batch[TEXT_LOGITS_S2S]
            tgt_mask, tgt_key_padding_mask = batch[TGT_MASK], batch[TGT_KEY_PADDING_MASK]
            text_target = batch[TEXT]
        logits_len = batch[UNPADDED_TEXT_LEN]

        out = self.inference(x, width, max_char_len=logits_target_s2s.shape[-1]+10)

        pred_text_s2s, _ = self.alphabet.batch_logits_to_string_list(torch.argmax(out["predictions_s2s"].cpu(), dim=-1),
                                                                     stopping_logits=[
                                                                         self.alphabet.string_to_logits(
                                                                             END_OF_SEQUENCE)])
        self.cer_s2s_test(pred_text_s2s, text_target)
        self.wer_s2s_test(pred_text_s2s, text_target)

        self.test_pred_text_s2s = pred_text_s2s
        self.test_target_text = text_target

        if self.transcription_target == "dipl":
            pred_abbr = extract_abbreviations(pred_text_s2s, EX_OPEN, EX_CLOSE)
            tgt_abbr = extract_abbreviations(text_target, EX_OPEN, EX_CLOSE)
            self.cer_s2s_abbr_test(pred_abbr, tgt_abbr)
            self.wer_s2s_abbr_test(pred_abbr, tgt_abbr)

    def on_test_epoch_end(self):
        cer_results = self.cer_s2s_test.compute()
        wer_results = self.wer_s2s_test.compute()
        print("CER", cer_results, "WER", wer_results)
        self.log("test/cer_s2s", cer_results)
        self.log("test/wer_s2s", wer_results)
        self.cer_s2s_test.reset()
        self.wer_s2s_test.reset()
        cer_abbr_results = self.cer_s2s_abbr_test.compute()
        wer_abbr_results = self.wer_s2s_abbr_test.compute()
        print("CER ABBR", cer_abbr_results, "WER ABBR", wer_abbr_results)
        self.log("test/cer_s2s_abbr", cer_abbr_results)
        self.log("test/wer_s2s_abbr", wer_abbr_results)
        self.cer_s2s_abbr_test.reset()
        self.wer_s2s_abbr_test.reset()

    def optimizer_step(self, epoch, batch_idx, optimizer, opt_closure):
        if self.trainer.global_step < self.hparams.warm_up_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warm_up_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=opt_closure)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler is None:
            return optimizer
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.metric,
                "interval": "epoch",
                "frequency": 1,
            }
        }
