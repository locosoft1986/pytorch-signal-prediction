import torch
import torch.nn as nn
import torch.autograd as autograd

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

        self.linear = nn.Linear(hidden_size, output_size)
        self.scale = autograd.Variable(torch.FloatTensor([1]), requires_grad = True)

    def forward(self, x, hidden):
        decoder_output, next_hidden = self.decoder(x, hidden)
        # use the last batch of outputs, so it should be [:, -1, :]
        linear_output = self.linear(decoder_output[:, -1, :])
        output = self.scale.expand_as(linear_output) * linear_output

        return output, decoder_output, next_hidden