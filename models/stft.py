import torch
import torch.nn as nn


class StftTransform(nn.Module):
    def __init__(self,
                 n_fft=1024,  # 512로 줄임
                 hop_length=256,  # 128로
                 win_length=1024
                 ):
        super(StftTransform, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, noisy_wave, clean_wave):
        window = torch.hann_window(self.win_length).to(x.device)

        noisy_wave = noisy_wave.squeeze(1)   # (batch_size, 1, 288000) -> (batch_size,288000)
        clean_wave = clean_wave.squeeze(1)
        # STFT 변환
        noisy_spec = torch.stft(
            noisy_wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        )
        clean_spec = torch.stft(
            clean_wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        )

        noisy_spec = noisy_spec.unsqueeze(1)  # (batch_size, 1, 513, 1126)
        clean_spec = clean_spec.unsqueeze(1)

        spec_mag = abs(noisy_spec)
        return spec_mag, noisy_spec, clean_spec
