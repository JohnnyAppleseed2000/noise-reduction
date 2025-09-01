import torch.nn as nn


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
        # Skip-connection 위해 각 Cnn Block 결과 저장
        encoder1 = self.cnn_encoder1(spec_mag)
        encoder2 = self.cnn_encoder2(encoder1)
        encoder3 = self.cnn_encoder3(encoder2)
        x = encoder3

        # 메모리 절약
        encoder1 = encoder1.detach()
        encoder2 = encoder2.detach()
        encoder3 = encoder3.detach()

        return x, encoder1, encoder2, encoder3
