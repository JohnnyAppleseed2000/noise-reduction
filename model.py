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

    def forward(self, x):  # x:(1, 288000)를 stft 통해 (1, 513, 1126)로 변환
        window = torch.hann_window(self.win_length).to(x.device)

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
        encoder1 = self.cnn_encoder1(spec_mag)  # Skip-connection 위해 각 Cnn Block 결과 저장
        encoder2 = self.cnn_encoder2(encoder1)
        encoder3 = self.cnn_encoder3(encoder2)
        x = encoder3

        return x, encoder1, encoder2, encoder3


class Transformer(nn.Module):
    def __init__(self,
                 num_layers=4,
                 d_model=512,
                 num_head=8
                 ):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_head = num_head

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_head), num_layers=self.num_layers
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
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()  # (1, 16, 260, 564) -> (1, 1, 520, 1128)
        )

    def forward(self, x, encoder1, encoder2, encoder3):
        # Block1: [Skip connection -> Upsampling]
        decoder1 = self.cnn_decoder1(torch.cat((x, encoder3), dim=1))
        encoder2_resized = F.interpolate(encoder2, size=(130, 282),
                                         mode='bilinear',
                                         align_corners=False)
        # Block2: [Skip connection -> Upsampling]
        decoder2 = self.cnn_decoder2(torch.cat((decoder1, encoder2_resized), dim=1))
        encoder1_resized = F.interpolate(encoder1, size=(260, 564),
                                         mode='bilinear',
                                         align_corners=False)
        # Block3: [Skip connection -> Upsampling -> 마지막 출력을 입력과 같은 사이즈로 복원]
        decoder3 = self.cnn_decoder3(torch.cat((decoder2, encoder1_resized), dim=1))
        decoder3_resized = F.interpolate(decoder3, size=(513, 1126),
                                         mode='bilinear',
                                         align_corners=False)
        return decoder3_resized


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.stft = Stft()
        self.encoder = CnnEncoder()
        self.transformer = Transformer()
        self.decoder = CnnDecoder()
        self.pre_proj = nn.Linear(in_features=2080, out_features=512)
        self.post_proj = nn.Linear(in_features=512, out_features=2080)

    def forward(self, x):
        x, spec = self.stft(x)  # (batch_size, channel, freq, time): (batch_size, 1, 513, 1126)
        x, encoder1, encoder2, encoder3 = self.encoder(x)  # (batch_size, 32, 65, 141)

        (batch_size, channels, freq, time) = x.shape
        x = x.reshape(time, batch_size, freq * channels)  # (65, batch_size, 2080)

        x = self.pre_proj(x)  # transformer 의 차원에 맞춰주기 위해 linear projection -> (65, batch_size, 512)
        x = self.transformer(x)
        x = self.post_proj(x)  # (65, batch_size, 512) -> (65, batch_size, 2080)

        x = x.reshape(batch_size, channels, freq, time)  # (batch_size, 32, 65, 141)
        x = self.decoder(x, encoder1, encoder2, encoder3)
        return x
