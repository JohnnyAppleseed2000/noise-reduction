import torch
import torch.nn as nn
from .cnn_decoder import CnnDecoder
from .cnn_encoder import CnnEncoder
from .transformer import Transformer
from .stft import StftTransform
from .positional_encoding import PositionalEncoding

class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.stft_transform = StftTransform()
        self.encoder = CnnEncoder()
        self.transformer = Transformer()
        self.decoder = CnnDecoder()
        self.pre_proj = nn.Linear(in_features=2080, out_features=512)
        self.post_proj = nn.Linear(in_features=512, out_features=2080)
        self.positional_encoding = PositionalEncoding(d_model=512, max_len=1000)

    def forward(self, noisy_wave, clean_wave):
        #  == 1. STFT 변환 ==
        x, noisy_spec, clean_mag = self.stft_transform(noisy_wave, clean_wave)

        #  == 2. CNN Encoder ==
        x, encoder1, encoder2, encoder3 = self.encoder(x)  # (batch_size, 32, 65, 141)

        # == 3. Transformer ==
        (batch_size, channels, freq, time) = x.shape
        x = x.reshape(time, batch_size, freq * channels)  # (65, batch_size, 2080)
        x = self.pre_proj(x)  # transformer 의 차원에 맞춰주기 위해 linear projection -> (65, batch_size, 512)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.post_proj(x)  # (65, batch_size, 512) -> (65, batch_size, 2080)

        # == 4. CNN Decoder ==
        x = x.reshape(batch_size, channels, freq, time)  # (batch_size, 32, 65, 141)
        x = self.decoder(x, encoder1, encoder2, encoder3)
        del encoder1, encoder2, encoder3

        # == 5. 위상 복원 ==
        recon_noisy = x * torch.exp(1j * torch.angle(noisy_spec))
        del noisy_spec

        return recon_noisy, x, clean_mag

