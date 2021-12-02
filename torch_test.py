import torch
import torch.nn as nn

seq_size, batch_size, input_size = 5, 4, 20
input_size, hidden_size, num_layers = 20, 16, 3
i, h, n = input_size, hidden_size, num_layers

t = torch.rand(seq_size, batch_size, input_size)
print(t)

# input, hidden, n_layer
rnn = torch.nn.RNN(i, h, n, bidirectional=False) 
rnn2 = torch.nn.RNN(i, h, n, bidirectional=True) 
# param batch_first?

result = rnn(t)
result2 = rnn2(t)

print(result)

# (output, hidden)
print(result[0].size(), result[1].size())   # (5,4,16), (3,4,16)
print(result2[0].size(), result2[1].size()) # (5,4,32), (6,4,16)

# output : seq, batch, input_size -> hidden_size (or twice if bidirectional)
# hidden : n_layers(or twice if bidirectional), batch_size, hidden_size