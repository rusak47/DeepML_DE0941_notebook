import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import json

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

plt.rcParams["figure.figsize"] = (10, 16) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 32

EMBEDDING_SIZE = 32
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 2
RNN_IS_BIDIRECTIONAL = False

TRAIN_TEST_SPLIT = 0.8

MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    # comment out this next line if you have nvidia GPU and you want to debug
    MAX_LEN = None
    pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/quotes.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/quotes.pkl',
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
        x_raw = np.array(self.final_quotes_sentences[idx], dtype=np.int64) # A, B, C, D

        # x - to be prepared is half the victory (A,B,C)
        # y - be prepared is half the victory (B, C, D)
        y = np.roll(x_raw, -1) # [1,2,3,4] =becomes> [2,3,4,1]
        y = y[:-1] # cut last
        x = x_raw[:-1] # [1,2,3]

        x_len = len(x)
        pad_right = self.max_sentence_length - x_len
        x_padded = np.pad(x, (0, pad_right))
        y_padded = np.pad(x, (0, pad_right))

        # TODO
        return x_padded, y_padded, x_len

dataset_full = Dataset()

train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1), ## its important to drop last if 1 batch norm will fail because cant count  mean and sigma for one sample
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1),
    shuffle=False

)


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #h_t = tahn(W @ h_<t-1> + U@x_t + b)
        # o_t = V @ h_t + b_o
        #parameter is simple matrix
        self.W = torch.nn.Linear( # allows to call forwards pass directly
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            #torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U = torch.nn.Linear(
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            #torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )
        self.V = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=len(dataset_full.vocabulary_keys)
            #torch.FloatTensor(RNN_HIDDEN_SIZE, len(dataset_full.vocabulary_keys))
        )

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_len = pad_packed_sequence(x, batch_first=True) # True <- makes longest sentence first (safety switch)
        if hidden is None:
            hidden = torch.zeros(x_unpacked.size(0), RNN_HIDDEN_SIZE).to(DEVICE) # why its a bad idea to put into default parameter - its a pointer that points to heap
                                                                                # fn used globally will work with defaut value

        # x_unpacked B, Seq, F
        x_seq = x_unpacked.permute(1,0,2)
        outs = []
        for x_t in x_seq: # (B,F)
            W_dot_x = self.W.forward(hidden)
            U_dot_x = self.U.forward(x_t)
            hidden = torch.tanh_((W_dot_x + U_dot_x))
            out = self.V.forward(hidden)
            outs.append(out)

        out_seq = torch.stack(outs)

        out_seq = out_seq.permute(1,0,2) # Seq, B, F -> B. Seq, F

        output = pack_padded_sequence(out_seq, x_len, batch_first=True, enforce_sorted=False)

        return output, hidden

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #embeds
        self.emb = torch.nn.Embedding(
            num_embeddings= len(dataset_full.vocabulary_keys),
            embedding_dim=EMBEDDING_SIZE
        )

        self.rnn = RNN()

    def forward(self, x: PackedSequence, hidden=None):
        # B,Seq, 1 -> B, Seq, Emb
        x_emb = self.emb.forward(x.data) # x.dat is the sousage of all sentences (packedsequenceofsentences)
        x_emb_packed = PackedSequence(
            data=x_emb,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        out, hidden =  self.rnn.forward(x_emb_packed,hidden)
        y_prim = torch.softmax(out.data, dim=-1) # sousage probabs of words

        y_prim_packed = PackedSequence(
            data = y_prim,
            # tokens where we need to cut the sousage
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        return y_prim_packed, hidden


model = Model()
model = model.to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_weights = torch.FloatTensor( 1./ np.array((dataset_full.vocabulary_counts))) # inverse a/counts, a - multiplier to increase
# instead dataset_full, should be dataset_train
loss_weights = loss_weights.to(DEVICE)

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []

fig, (ax1, ay1) = plt.subplots(2,1)
plt.ion()

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        for x_padded, y_padded, x_length in dataloader:

            x_padded = x_padded.to(DEVICE)
            y_padded = y_padded.to(DEVICE)

            x_packed = pack_padded_sequence(x_padded, x_length, batch_first=True, enforce_sorted=False)
            y_packed = pack_padded_sequence(y_padded, x_length, batch_first=True, enforce_sorted=False)

            y_prim_packed, _ = model.forward(x_packed)

            idxes_batch = range(len(y_packed.data))
            idxes_y = y_packed.data
            #L_cce = -y*log(y') = -log(y'_<y_idx>) < doing cross entropy for the specific y_idx
            loss = -torch.mean(loss_weights[idxes_y] * torch.log(y_prim_packed.data[idxes_batch, idxes_y] + 1e-8))
            losses.append(loss.cpu().item())

            idxes_y_prim = y_prim_packed.data.argmax(dim=-1)
            acc = torch.mean((idxes_y_prim == idxes_y) * 1.0)
            accs.append(acc.cpu().item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))

    print(
        f'\n\nepoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_plot_train: {acc_plot_train[-1]} '
        f'acc_plot_test: {acc_plot_test[-1]} '
    )

    if epoch % 10 == 0 or True:
        plt.clf()
        ax1.plot(loss_plot_train, 'r-', label='loss train')
        ax1.legend()
        ax1.set_xlabel("Epoch")
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='loss test')
        ax2.legend(loc='upper left')


        ay1.plot(acc_plot_train, 'r-', label='acc train')
        ay1.legend()
        ay1.set_xlabel("Epoch")
        ay2 = ay1.twinx()
        ay2.plot(acc_plot_test, 'c-', label='acc test')
        ay2.legend(loc='upper left')

        plt.tight_layout(pad=0.5)
        plt.draw()
        plt.pause(0.1)

        # rollout code - text generator
        x_roll = x_padded[:, :1]
        hidden = None
        batch_size = x_roll.size(0)

        for t in range(dataset_full.max_sentence_length):
            x_packed = pack_padded_sequence(
                x_roll[:, -1:],
                lengths=torch.LongTensor([1] * batch_size),  # [1, 1, 1, 1 ...]
                batch_first=True,
            ).to(DEVICE)

            y_prim, hidden = model.forward(x_packed, hidden)

            y_prim_unpacked, _ = pad_packed_sequence(y_prim, batch_first=True)
            y_prim_idx = y_prim_unpacked.argmax(dim=-1)
            x_roll = torch.cat((x_roll, y_prim_idx), dim=-1).to(DEVICE)

        np_x_roll = x_roll.cpu().numpy()
        for sent in np_x_roll:
            words = [dataset_full.vocabulary_keys[it] for it in sent]
            if '[eos]' in words:
                eos_idx = words.index('[eos]')
                words = words[:eos_idx]
            print(' '.join(words))