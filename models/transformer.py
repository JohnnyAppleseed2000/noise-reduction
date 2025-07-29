import torch.nn as nn


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
