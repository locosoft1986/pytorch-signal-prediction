import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 number_of_layers=1,
                 dropout=0,
                 bidirectional=False):

        super(Decoder, self).__init__()

        self.decoder = nn.LSTM(input_size,
                               hidden_size,
                               num_layers=number_of_layers,
                               dropout=dropout,
                               bidirectional=bidirectional)

        self.linear = nn.Linear(recurrent_layer_size, output_size)

    def forward(self, x):

        decoder_output, _ = self.decoder(x)

        linear_output = self.linear(decoder_output[:, -1, :])

        return linear_output