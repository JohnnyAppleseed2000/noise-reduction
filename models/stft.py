import torch
import torch.nn as nn

def specnorm(noisy_wave, clean_wave):


class StftTransform(nn.Module):
    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 noisy_mean=0.3172,
                 noisy_std=1.4614,
                 clean_mean=0.1690,
                 clean_std=1.1905
                 ):
        super(StftTransform, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.noisy_mean = noisy_mean
        self.noisy_std = noisy_std
        self.clean_mean = clean_mean
        self.clean_std = clean_std
        self.register_buffer('window', torch.hann_window(self.win_length))

    def forward(self, noisy_wave, clean_wave):  # 입력:(batch_size, 1, 288000)
        window = self.window.to(noisy_wave.device)
        # 1. STFT는 2차원 이하의 입력을 받기 때문에 squeeze를 통해 (batch_size, 1, 288000) -> (batch_size, 288000)
        noisy_wave = noisy_wave.squeeze(1)
        clean_wave = clean_wave.squeeze(1)

        # 2. STFT 변환 -> (batch_size, 513, 1126)
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

        # 3. Unsqueeze 통해 채널 차원 추가 -> (batch_size, 1, 513, 1126)
        noisy_spec = noisy_spec.unsqueeze(1)
        clean_spec = clean_spec.unsqueeze(1)

        # 4. 이전에 구한 global mean, std 를 통해 Normalization 진행
        noisy_mag = torch.abs(noisy_spec)
        noisy_mag = torch.log1p(noisy_mag)
        noisy_mag = (noisy_mag - self.noisy_mean) / self.noisy_std
        clean_mag = torch.abs(clean_spec)
        clean_mag = torch.log1p(clean_mag)
        clean_mag = (clean_mag - self.clean_mean) / self.clean_std
        return noisy_mag, noisy_spec, clean_mag

