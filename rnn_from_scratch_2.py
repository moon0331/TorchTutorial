import unicodedata
import string
import glob
import os

import torch
from torch import nn
import torch.nn.functional as F

DIR_NAME = 'data/names/*.txt'

ALL_LETTERS = string.ascii_letters + " .,;'"

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