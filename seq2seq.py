import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

UNK = '<unk>'

tok = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter, tok), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# print(vocab['my'])
# print(vocab(['this', 'is', 'my', 'car']))
# print(vocab['<unk>']) #0

text_pipeline = lambda x: vocab(tok(x))
label_pipeline = lambda x: int(x) - 1

# print(text_pipeline('this is my car'))
# print(vocab(['this is my car']))
# print(tok('this is my car'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label)) # 0 ~ 3
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64) # int list to tensor
        text_list.append(processed_text) # tensor of int-list-tensor
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

collate_batch(train_iter)