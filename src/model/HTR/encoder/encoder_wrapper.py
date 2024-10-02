import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.model.HTR.encoder.HTRTransformerEncoder import HTRTransformerEncoder

class EncoderWrapper(nn.Module):
    def __init__(
            self,
            encoder,
            in_channels,
    ):
        super().__init__()
        if isinstance(encoder.func, type) and issubclass(encoder.func, nn.LSTM):
            self.out_dim = encoder.keywords["hidden_size"] * 2 if encoder.keywords["bidirectional"] else encoder.keywords["hidden_size"]
            self.encoder = encoder(
                input_size=in_channels,
            )
        elif isinstance(encoder.func, type) and issubclass(encoder.func, HTRTransformerEncoder):
            self.out_dim = encoder.keywords["d_model"]
            self.encoder = encoder(
                in_channels=in_channels,
            )
        elif isinstance(encoder.func, type) and issubclass(encoder.func, nn.Linear):
            raise NotImplementedError
        elif isinstance(encoder.func, type) and issubclass(encoder.func, nn.Identity):
            self.encoder = encoder()
            self.out_dim = in_channels
        else:
            raise NotImplementedError

    def forward(
            self,
            x,
            lengths=None,
    ):
        """
        :param x: batch_size x seq_len x in_dim
        :param lengths: batch_size
        :return:
        """
        if isinstance(self.encoder, nn.LSTM):
            if lengths is None:
                lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.int64)
            x = pack_padded_sequence(
                input=x,
                lengths=lengths,
                enforce_sorted=False,
                batch_first=True,
            )
            packed_out, _ = self.encoder(x)
            out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
            )
            return out
        elif isinstance(self.encoder, HTRTransformerEncoder):
            src_key_padding_mask = None
            if lengths is not None:
                src_key_padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
                for idx, w in enumerate(lengths):
                    src_key_padding_mask[idx, w:] = True
            return self.encoder(
                x=x,
                src_key_padding_mask=src_key_padding_mask,
            )
        elif isinstance(self.encoder, nn.Linear):
            raise NotImplementedError
        elif isinstance(self.encoder, nn.Identity):
            return self.encoder(x)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    from functools import partial
    from DAtoolbox.model.HTR.encoder.HTRTransformerEncoder import HTRTransformerEncoder
    encoder = partial(
        nn.LSTM,
        num_layers=2,
        bidirectional=True,
        batch_first=True,
        hidden_size=256,
    )
    # encoder = partial(
    #     HTRTransformerEncoder,
    # )
    # encoder = partial(
    #     nn.Linear,
    # )
    N = 3
    D = 64
    m = EncoderWrapper(encoder=encoder, in_channels=D)
    x = torch.randn(3, 15, D)
    lengths = torch.randint(1, 15, size=(3,))
    print(x.shape, lengths)
    print(m(x).shape)