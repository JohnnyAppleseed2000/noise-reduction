import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, spec_mag):
        x1 = self.cnn_encoder1(spec_mag)  # Skip-connection 위해 각 Cnn Block 결과 저장
        x2 = self.cnn_encoder2(x1)
        x = self.cnn_encoder3(x2)

        skip1 = x1
        skip2 = x2
        skip3 = x
        return x, skip1, skip2, skip3


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
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()  # (1, 64, 65, 141) -> (1, 16, 130, 282)
        )
        self.cnn_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()  # (1, 32, 130, 282) -> (1, 8, 260, 564)
        )
        self.cnn_decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16,1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()  # (1, 16, 260, 564) -> (1, 1, 520, 1128)
        )

    def forward(self, input, skip1, skip2, skip3):
        concat1 = torch.cat((input, skip3), dim=1)
        decoder1 = self.cnn_decoder1(concat1)
        concat2 = torch.cat((decoder1, skip2), dim=1)
        decoder2 = self.cnn_decoder2(concat2)
        concat3 = torch.cat((decoder2, skip1), dim=1)
        decoder3 = self.cnn_decoder3(concat3)
        return decoder3


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel,self).__init__()
        self.stft = Stft()
        self.encoder = CnnEncoder()
        self.transformer = Transformer()
        self.decoder = CnnDecoder()
        self.pre_proj = nn.Linear(in_features=2080, out_features=512)
        self.post_proj = nn.Linear(in_features=512, out_features=2080)

    def forward(self, x):
        x, spec = self.stft(x)  # (batch, channel, freq, time): (1, 1, 513, 1126)
        x, skip1, skip2, skip3 = self.encoder(x)  # (batch, channels, freq, time): (1, 32, 65, 141)

        (batch, channels, freq, time) = x.shape
        x = x.reshape(batch, time, freq*channels)  # (1, 65, 2080)
        x = x.permute(1, 0, 2)

        x = self.pre_proj(x)  # transformer 입력 맞추기 위해 (1, 141, 2080) --> (1, 141, 512)
        x = self.transformer(x)
        x = self.post_proj(x)  # (1, 141, 512) -> (1,141, 2080)

        x = x.reshape(batch, channels, freq, time)  # (1, 32, 65, 141)
        x = self.decoder(x, skip1, skip2, skip3)
        x = F.interpolate(x, size=(513, 1126), mode='bilinear', align_corners=False)
        return x



