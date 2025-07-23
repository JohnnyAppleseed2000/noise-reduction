import torch
import torch.nn as nn


class Stft(nn.Module):
    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024
                 ):
        super(Stft, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x):  # x:(1, 288000)를 stft 통해 (1, 513, 1125)로 변환
        window = torch.hann_window(self.win_length)

        # STFT 변환
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        )
        spec_mag = abs(spec)
        return spec_mag, spec


class CnnEncoder(nn.Module):
    def __init__(self):
        super(CnnEncoder, self).__init__()

        # CNN Encoder
        self.cnn_encoder1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()  # (1, 513, 1126) --> (8, 257, 563)
        )
        self.cnn_encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()  # (8, 257, 563) --> (16, 129, 282)
        )
        self.cnn_encoder3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (16, 129, 282) --> (32, 65, 141)
        )
        self.proj = nn.Linear(in_features=2080, out_features=512)

    def forward(self, spec_mag):
        x1 = self.cnn_encoder1(spec_mag)  # Skip-connection 위해 각 Cnn Block 결과 저장
        x2 = self.cnn_encoder2(x1)
        x3 = self.cnn_encoder3(x2)
        (batch, channels, freq, time_step) = x3.shape()
        x = x3.permute(0, 3, 2, 1)
        x = x.reshape(batch, time_step, freq*channels)
        x = self.proj(x)  # transformer 입력 크기 맞추기 위해 (1, 141, 2080) --> (1, 141, 512)로 linear projection
        return x, x1, x2, x3, time_step, freq, channels


class Transformer(nn.Module):
    def __init__(self,
                 num_layers=4,
                 d_model=512,
                 nhead=8
                 ):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead), num_layers=self.num_layers
        )

    def forward(self, x):
        x = self.transformer(x)

        return x


class CnnDecoder(nn.Module):
    def __init__(self):
        super(CnnDecoder, self).__init__()

        # CNN Decoder
        self.cnn_decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.cnn_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(48, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.cnn_decoder3 = nn.Sequential(
            nn.ConvTranspose2d(24,8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

    def forward(self, x, x1, x2, x3):
        skip_connection1 = torch.cat((x, x3), dim=1)
        decoder1 = self.cnn_decoder1(skip_connection1)
        skip_connection2 = torch.cat((decoder1, x2), dim=1)
        decoder2 = self.cnn_decoder2(skip_connection2)
        skip_connection3 = torch.cat((decoder2, x1), dim=1)
        decoder3 = self.cnn_decoder3(skip_connection3)
        return decoder3


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel,self).__init__()
        self.stft = Stft()
        self.encoder = CnnEncoder()
        self.transformer = Transformer()
        self.decoder = CnnDecoder()

    def forward(self, x):
        x, spec = self.stft(x)
        x, x1, x2, x3, time_step, freq, channels = self.encoder(x)
        x = self.transformer(x)
        Hy65
        x = self.decoder(x, x1, x2, x3)
        return x



