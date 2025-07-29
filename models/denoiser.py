import torch
import torch.nn as nn
from .cnn_decoder import CnnDecoder
from .cnn_encoder import CnnEncoder
from .transformer import Transformer
from .stft import StftTransform


class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.stft_transformer = StftTransform()
        self.encoder = CnnEncoder()
        self.transformer = Transformer()
        self.decoder = CnnDecoder()
        self.pre_proj = nn.Linear(in_features=2080, out_features=512)
        self.post_proj = nn.Linear(in_features=512, out_features=2080)

    def forward(self, x):
        x, noisy_spec, clean_spec = self.stft_transform(x)
        x, spec = self.stft(x)  # (batch_size, channel, freq, time): (batch_size, 1, 513, 1126)
        x, encoder1, encoder2, encoder3 = self.encoder(x)  # (batch_size, 32, 65, 141)

        (batch_size, channels, freq, time) = x.shape
        x = x.reshape(time, batch_size, freq * channels)  # (65, batch_size, 2080)

        x = self.pre_proj(x)  # transformer 의 차원에 맞춰주기 위해 linear projection -> (65, batch_size, 512)
        x = self.transformer(x)
        x = self.post_proj(x)  # (65, batch_size, 512) -> (65, batch_size, 2080)

        x = x.reshape(batch_size, channels, freq, time)  # (batch_size, 32, 65, 141)
        x = self.decoder(x, encoder1, encoder2, encoder3)

        # Clean 한 음성의 spectogram과 비교를 위해 복소수 성분 더해줌
        recon_noisy = x * torch.exp(1j * torch.angle(noisy_spec))
        return recon_noisy, clean_spec
