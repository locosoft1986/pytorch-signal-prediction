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
        
    def forward(self, x, hidden):
        decoder_output, next_hidden = self.decoder(x, hidden)
        outputs = []
        for i in range(decoder_output.size()[1]):
            outputs += [self.linear(decoder_output[:, i, :])]
        
        return torch.stack(outputs, dim=1).squeeze(), decoder_output, next_hidden
    