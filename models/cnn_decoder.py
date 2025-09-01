import torch
import torch.nn as nn
import torch.nn.functional as F


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
        del decoder1, encoder2_resized
        encoder1_resized = F.interpolate(encoder1, size=(260, 564),
                                         mode='bilinear',
                                         align_corners=False)

        # Block3: [Skip connection -> Upsampling -> 마지막 출력을 입력과 같은 사이즈로 복원]
        decoder3 = self.cnn_decoder3(torch.cat((decoder2, encoder1_resized), dim=1))
        del decoder2, encoder1_resized
        decoder3_resized = F.interpolate(decoder3, size=(513, 1126),
                                         mode='bilinear',
                                         align_corners=False)
        del decoder3
        return decoder3_resized