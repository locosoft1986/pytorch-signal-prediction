import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 number_of_layers=1,
                 dropout=0,
                 bidirectional=False):

        super(Encoder, self).__init__()

        self.encoder = nn.LSTM(input_size,
                               hidden_size,
                               num_layers=number_of_layers,
                               dropout=dropout,
                               bidirectional=bidirectional)

    def forward(self, x, hidden):

        encoder_output, encoder_state = self.encoder(x, hidden)

        return encoder_output, encoder_state