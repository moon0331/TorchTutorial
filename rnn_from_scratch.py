# RNN으로 학습

# data/names/*.txt : * 언어의 이름나열

# 이름 - 레이블 학습

import glob
import os
import string
import unicodedata

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

DIR_NAME = 'data/names/*.txt'

ALL_LETTERS = string.ascii_letters + " .,;'"


class RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.n_hidden = n_hidden

        self.i2o = nn.Linear(n_input+n_hidden, n_output)
        self.i2h = nn.Linear(n_input+n_hidden, n_hidden)

    def forward(self, inpt, hidn):
        concat = torch.cat((inpt, hidn), 1)
        output = self.i2o(concat)
        hidden = self.i2h(concat)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.n_hidden)


def unicodeToAscii(s):
    return ''.join(
        c.lower() for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

if __name__ == '__main__':

    lang_data = {}

    whole_files = glob.glob(DIR_NAME)
    n_file = len(whole_files)
    idx_to_language = []

    for filename in whole_files:
        lang = os.path.splitext(os.path.basename(filename))[0]
        idx_to_language.append(lang)
        lang_data[lang] = [
            unicodeToAscii(line.strip()) for line in open(filename, 'r')
        ]

    idx_to_language.sort()
    language_to_idx = {lang:i for i, lang in enumerate(idx_to_language)}
    n_language = len(idx_to_language)

    training_data = torch.Tensor()

    lang_onehot = F.one_hot(torch.arange(n_language))
    char_onehot = F.one_hot(torch.arange(len(ALL_LETTERS)))

    n_input = len(ALL_LETTERS)
    n_hidden = 128
    n_output = n_language

    x_idx_list = []
    y_idx_list = []

    for lang, words in lang_data.items():
        for word in words:
            x_idx_list.append(torch.tensor([ALL_LETTERS.index(c) for c in word]))
            y_idx_list.append(torch.tensor([language_to_idx[lang]]))

    epochs = 1000

    rnn = RNN(n_input, n_hidden, n_output)
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.005
    momentum = 0.9
    opt = optim.SGD(rnn.parameters(), lr=lr)

    bs = 64

    n_train = 100000
    plot_every = 500
    avg_loss = 0
    for epoch in range(n_train):
        rnd = torch.randint(0, len(x_idx_list), (1,))
        x = x_idx_list[rnd]
        y = y_idx_list[rnd]
        inputs = torch.stack([char_onehot[idx] for idx in x])
        inputs = torch.unsqueeze(inputs, 1)
        # output = lang_onehot[y]

        hidden = rnn.initHidden()

        # print(inputs.size(), output.size())
        for inpt in inputs:
            rnn_output, hidden = rnn(inpt, hidden)

        loss = loss_fn(rnn_output, y) #?

        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_loss += loss/plot_every

        if epoch and epoch % plot_every == 0:
            print(f'avg loss = {avg_loss:5f}, ', end='')
            avg_loss = 0

            with torch.no_grad():
                correct = 0
                for i in range(len(x_idx_list)):
                    x = x_idx_list[i]
                    y = y_idx_list[i]
                    inputs = torch.stack([char_onehot[idx] for idx in x])
                    inputs = torch.unsqueeze(inputs, 1)
                    hidden = rnn.initHidden() # 이거 빼먹지 말기
                    for inpt in inputs:
                        rnn_output, hidden = rnn(inpt, hidden)

                    if rnn_output.argmax().item() == y.item():
                        correct += 1

                print(f'accuracy = {correct / len(x_idx_list) :5f}')

# 