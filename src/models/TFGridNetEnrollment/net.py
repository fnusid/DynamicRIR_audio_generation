import torch
import torch.nn as nn

from .tfgridnet import TFGridNet
import torch.nn.functional as F
from  scipy.signal.windows import tukey


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class Net(nn.Module):
    def __init__(self, stft_chunk_size=64, stft_pad_size = 32, stft_back_pad = 32,
                 num_ch=2, D=64, B=6, I=1, J=1, L=0, H=128, local_atten_len=100,
                 E = 4, chunk_causal=False, num_src = 2, spk_emb_dim=256,
                 spectral_masking=False, use_first_ln=False, merge_method = "None",
                 conv_lstm = True, lstm_down=5, use_attn=False, masked_attn = False):
        super(Net, self).__init__()
        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.num_ch = num_ch
        self.stft_back_pad = stft_back_pad
        self.layers = B
        self.n_srcs = num_src

        self.embed_dim = D
        self.E = E

        # Input conv to convert input audio to a latent representation
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size
        
        self.nfreqs = self.nfft//2 + 1

        # Construct synthesis/analysis windows
        self.analysis_window = torch.from_numpy( tukey(self.nfft, alpha = ((stft_pad_size+stft_back_pad) / 2) /self.nfft) ).float()

        # TF-GridNet        
        self.tfgridnet = TFGridNet(n_srcs=num_src,
                                   spk_emb_dim=spk_emb_dim,
                                   n_fft=self.nfft,
                                   stride=stft_chunk_size,
                                   emb_dim=D,
                                   emb_ks=I,
                                   emb_hs=J,
                                   n_layers=self.layers,
                                   n_imics=num_ch,
                                   attn_n_head=L,
                                   attn_approx_qk_dim=E*self.nfreqs,
                                   lstm_hidden_units=H,
                                   local_atten_len=local_atten_len,
                                   chunk_causal = chunk_causal,
                                   spectral_masking = spectral_masking,
                                   use_first_ln = use_first_ln,
                                   merge_method = merge_method,
                                   conv_lstm = conv_lstm,
                                   lstm_down = lstm_down,
                                   masked_attn = masked_attn)

    def extract_features(self, x):
        """
        x: (B, M, T)
        returns: (B, M, C*F, T)
        """
        B, M, T = x.shape

        x = x.reshape(B*M, T)
        x = torch.stft(x, n_fft = self.nfft, hop_length = self.stft_chunk_size,
                          win_length = self.nfft, window=self.analysis_window.to(x.device),
                          center=False, normalized=False, return_complex=True)
        x = torch.view_as_real(x) # [B*M, F, T, C]

        x = x.permute(0, 3, 1, 2) # [B*M, C, F, T]
        
        BM, C, _F, T = x.shape
        x = x.reshape(B * M,  C * _F, T) # [BM, CF, T]
        x = x.reshape(B, M, C * _F, T) # [B, M, CF, T]

        return x

    def predict(self, x, embed, pad=True):
        """
        x: (B, M, t)
        """

        # Time-domain to TF-domain
        x = self.extract_features(x) # [B, M, CF, T]

        x = self.tfgridnet(x)

        return x

    def forward(self, inputs, pad=True):
        x = inputs['enrollment']

        embedding = self.predict(x, pad)

        return {'output': embedding}

if __name__ == "__main__":
    pass