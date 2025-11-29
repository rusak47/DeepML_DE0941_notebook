import json
import os
import pdb
import pickle

import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data
import torch.nn.functional as F
import argparse


HIDDEN_SIZE = 64
TRANSFORMER_LAYERS = 8
DROPOUT = 0.1
TRANSFORMER_HEADS = 4
DEVICE = 'cpu'
MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = 0

PATH_DATA = './results'
os.makedirs(PATH_DATA, exist_ok=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/quotes_small.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/quotes_small.pkl',
                path_dataset,
                progress=True
            )

        with open(path_dataset, 'rb') as fp:
            (
                self.final_quotes_sentences, self.final_authors, self.final_categories,
                self.vocabulary_keys, self.vocabulary_counts, self.authors_keys, self.categories_keys
            ) = pickle.load(fp)
        self.max_sentence_length = np.max([len(it) for it in self.final_quotes_sentences])

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.final_quotes_sentences)

    def __getitem__(self, idx):
        x_raw = np.array(self.final_quotes_sentences[idx], dtype=np.int64)

        y = np.roll(x_raw, -1) # move all words by one position left
        y = y[:-1]
        x = x_raw[:-1]
        x_length = len(x)

        pad_right = self.max_sentence_length - x_length
        pad_left = 0
        x_padded = np.pad(x, (pad_left, pad_right))
        y_padded = np.pad(y, (pad_left, pad_right))

        return x_padded, y_padded, x_length

dataset_full = Dataset()

class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, idxes):
        return self.pe[idxes, :]


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.project_k = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.project_q = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.project_v = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)

    def forward(self, x, lengths, atten):
        batch_size = x.size(0) # x.shape (B, seq, HIDDEN_SIZE)
        seq_size = x.size(1)

        # shape (B, seq, HIDDEN_SIZE)
        k = self.project_k.forward(x)
        q = self.project_q.forward(x)
        v = self.project_v.forward(x)

        # shape (B, seq, Heads, HIDDEN_SIZE/Heads)
        # shape (B, Heads, seq, HIDDEN_SIZE/Heads)
        k = k.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)
        q = q.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)
        v = v.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)

        atten_raw = q @ k.transpose(-1, -2) / np.sqrt(x.size(-1))

        mask = torch.tril(torch.ones(seq_size, seq_size)).to(DEVICE) # (Seq, Seq)
        atten_mask = atten_raw.masked_fill(mask == 0, value=float('-inf'))  # (B, H, Seq, Seq)
        for idx, length in enumerate(lengths): # (B, Seq, Seq)
            atten_mask[idx, :, :, length:] = float('-inf')
            atten_mask[idx, :, length:, :] = float('-inf')

        atten = torch.softmax(atten_mask, dim=-1)
        atten = atten.masked_fill(((atten > 0) == False), value=0.0)
        out = atten @ v

        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_size, HIDDEN_SIZE)
        atten = atten.detach().mean(dim=1) # shape (B, Heads, seq, seq) => (B, seq, seq)

        # torch.nn.Module > self.training
        # model.eval() model.train()
        out_1 = x + torch.dropout(out, p=DROPOUT, train=self.training)
        out_1_norm = self.norm_1.forward(out_1)

        out_2 = self.ff.forward(out_1_norm)
        out_3 = out_1_norm + out_2
        y_prim = self.norm_2.forward(out_3)

        return y_prim, lengths, atten


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.project_w_e = torch.nn.Embedding(
            num_embeddings=len(dataset_full.vocabulary_keys),
            embedding_dim=HIDDEN_SIZE
        )
        self.project_p_e = torch.nn.Embedding(
            num_embeddings=dataset_full.max_sentence_length,
            embedding_dim=HIDDEN_SIZE
        )

        self.transformer = torch.nn.ModuleList(
            [TransformerLayer() for _ in range(TRANSFORMER_LAYERS)]
        )

        self.fc = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, x: PackedSequence):

        x_e = PackedSequence(
            data=self.project_w_e.forward(x.data),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        x_e_unpacked, lengths = pad_packed_sequence(x_e, batch_first=True)

        # 0, 1, 2, 3, 4.... => (B, Seq) => project_p_e => (B, Seq, HIDDEN_SIZE)
        # lengths[0]
        pos_idxes = torch.arange(0, torch.max(lengths)).to(DEVICE)
        p_e = self.project_p_e.forward(pos_idxes) # (Seq,)
        p_e = p_e.unsqueeze(dim=0) # (1, Seq, H)
        p_e = p_e.expand(x_e_unpacked.size())

        z = x_e_unpacked + p_e
        atten = None
        for layer in self.transformer:
            z, lengths, atten = layer.forward(z, lengths, atten)

        z_packed = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        out_fc = self.fc.forward(z_packed.data)
        y_prim_logits = (self.project_w_e.weight @ out_fc.unsqueeze(dim=-1)).squeeze(dim=-1)
        y_prim = torch.softmax(y_prim_logits, dim=-1)

        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        return y_prim_packed, atten


model = Model()

if not os.path.exists(f'{PATH_DATA}/pretrain-model-259.pt'):
    download_url_to_file('http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/pretrain-model-259.pt', f'{PATH_DATA}/pretrain-model-259.pt', progress=True)

model.load_state_dict(
    state_dict=torch.load(
        f'{PATH_DATA}/pretrain-model-259.pt',
        map_location='cpu'
    )
)
model = model.train()
torch.set_grad_enabled(False)

while True:
    x_str = input('x: (exit with \q)')
    x_list = x_str.lower().split(' ')

    is_valid = True
    x_idxes = []
    for x_each in x_list:
        if x_each == '\q':
            exit()
        if x_each not in dataset_full.vocabulary_keys:
            print(f"\033[91m {x_each} not found in vocabulary \033[0m")
            is_valid = False
        else:
            x_idxes.append(dataset_full.vocabulary_keys.index(x_each))

    if not is_valid:
        continue

    # todo implement rollout (remember that the transformer has no memory.)
    # todo implement matplotlib attention matrix

    print('y_prim: ' + ' '.join(x_list))

