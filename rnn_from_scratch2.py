import unicodedata
import string
import glob
import os
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

DIR_NAME = 'data/names/*.txt'

ALL_LETTERS = string.ascii_letters + " .,;'"


class RNN(nn.Module):
    def __init__(self, n_category, n_words, n_hidden, n_output):
        super().__init__()
        self.n_hidden = n_hidden

        self.n_category = n_category
        self.n_words = n_words

        self.i2o = nn.Linear(n_category+n_words+n_hidden, n_output)
        self.i2h = nn.Linear(n_category+n_words+n_hidden, n_hidden)

        self.o2o = nn.Linear(n_hidden+n_output, n_output)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, t_category, t_input, t_hidden):
        combined = torch.cat((t_category, t_input, t_hidden), 1)
        output = self.i2o(combined)
        hidden = self.i2h(combined)
        out_combined = torch.cat((output, hidden), 1)
        output = self.o2o(out_combined)
        output = self.dropout(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.n_hidden)


def unicodeToAscii(s):
    return ''.join(
        c.lower() for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def getTrainingData(lang_data):
    langs_list = [key for key in lang_data]
    langs_list.sort()

    langs_dict = {lang:i for i, lang in enumerate(langs_list)}
    all_letters_dict = {c:ALL_LETTERS.index(c) for c in ALL_LETTERS}

    n_language = len(langs_list)
    n_letter = len(ALL_LETTERS) + 1 # add <EOS>

    full_language_tensor = F.one_hot(torch.arange(n_language)) # (18, 18)
    full_language_tensor = torch.unsqueeze(full_language_tensor, 1) # runtimeerror one of the variabels neede for grad...
    # full_language_tensor.unsqueeze_(1) # (18, 1, 18)
    full_char_tensor = F.one_hot(torch.arange(n_letter)) # (58, 58)
    full_char_tensor = torch.unsqueeze(full_char_tensor, 1)
    # full_char_tensor.unsqueeze_(1) #(58, 1, 58)

    lang_word_nextword_list = []
    for lang, words in lang_data.items():
        for word in words:
            lang_tensor = full_language_tensor[langs_dict[lang]] #(1, 18)
            word_input_tensors = torch.stack([full_char_tensor[all_letters_dict[c]] for c in word])
            word_output_tensors = torch.tensor([all_letters_dict[c] for c in word]+[n_letter-1])
            word_output_tensors = torch.unsqueeze(word_output_tensors, -1)
            # word_output_tensors.unsqueeze_(-1)
            lang_word_nextword_list.append((lang_tensor, word_input_tensors, word_output_tensors)) #(18), (4,58)

    return lang_word_nextword_list


def getRandomTrainingData(training_data):
    idx = random.randint(0, len(training_data)-1)
    return training_data[idx]

def sample(category, start_letter='A', lang_list=[], rnn=None):
    max_length = 20
    
    n_language = len(langs_list)
    n_letter = len(ALL_LETTERS) + 1 # add <EOS>

    with torch.no_grad():  # 샘플링에서 히스토리를 추적할 필요 없음
        category_idx = lang_list.index(category)
        category_tensor = F.one_hot(torch.arange(n_language))[category_idx]
        category_tensor = torch.unsqueeze(category_tensor, 0)
        input = ALL_LETTERS.index(start_letter)
        input = F.one_hot(torch.arange(n_letter))[input]
        input = torch.unsqueeze(input, 0)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input, hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letter - 1:
                break
            else:
                letter = ALL_LETTERS[topi]
                output_name += letter
            input = ALL_LETTERS.index(start_letter)
            input = F.one_hot(torch.arange(n_letter))[input]
            input = torch.unsqueeze(input, 0)

        return output_name

def samples(category, start_letters='ABC', langs_list=[], rnn=None):
    for start_letter in start_letters:
        print(sample(category, start_letter, langs_list, rnn))


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

    training_data = getTrainingData(lang_data)

    lr = 0.0005
    rnn = RNN(len(lang_data), len(ALL_LETTERS)+1, 64, len(ALL_LETTERS)+1)
    opt = optim.SGD(rnn.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    print(rnn)

    langs_list = [key for key in lang_data]
    langs_list.sort()

    total_loss = 0
    # epoch_loss = 0
    for epoch in range(1000):
        t_category, t_inputs, t_idx_outputs = getRandomTrainingData(training_data)
        # train()

        rnn.train()

        opt.zero_grad()

        t_hidden = rnn.initHidden()
        loss = 0
        for t_input, t_idx_output in zip(t_inputs, t_idx_outputs):
            output, hidden = rnn(t_category, t_input, t_hidden)
            loss += loss_fn(output, t_idx_output)

        loss.backward()
        # epoch_loss.backward(retain_graph=True) #https://prup.tistory.com/47
        opt.step()
        total_loss += loss

        if epoch and epoch%1000 == 0:
            print(f'epoch {epoch} : loss = {total_loss/1000:.5f}')
            total_loss = 0

            samples('Russian', 'RUS', langs_list, rnn)

            samples('German', 'GER', langs_list, rnn)

            samples('Spanish', 'SPA', langs_list, rnn)

            samples('Chinese', 'CHI', langs_list, rnn)

    # rnn.save()

# inference 다시 구현해보기