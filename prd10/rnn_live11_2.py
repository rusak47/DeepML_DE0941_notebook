import os
import pickle
import time
from warnings import catch_warnings

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

plt.rcParams["figure.figsize"] = (10, 16)  # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 32

EMBEDDING_SIZE = 32
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 2
RNN_IS_BIDIRECTIONAL = False #todo what is this

TRAIN_TEST_SPLIT = 0.8

MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'

SAVE_STATE_TEST_LOSS_SENSITIVITY = 3 # the greater the value, the more often weights will be refreshed

checkpoint_path = 'data/rnn_chpoints'
ch_file = 'rnn_best.pt'
USE_LSTM = False
USE_GRU = True
if USE_LSTM:
    ch_file = 'lstm_best.pt'
if USE_GRU:
    ch_file = 'gru_best.pt'


if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 1100 # full size 34k
    # MAX_LEN = None

print(f"maxlen: {MAX_LEN}; device: {DEVICE}")
print(f' using GRU:{USE_GRU}; LSTM:{USE_LSTM}; RNN:{not(USE_GRU or USE_LSTM)}')


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
        print(f' dataset real length: {len(self.final_quotes_sentences)}')
        if MAX_LEN:
            return MAX_LEN
        return len(self.final_quotes_sentences)

    def __getitem__(self, idx):
        x_raw = np.array(self.final_quotes_sentences[idx], dtype=np.int64)  # A, B, C, D

        # x - to be prepared is half the victory (A,B,C)
        # y - be prepared is half the victory (B, C, D)
        y = np.roll(x_raw, -1)  # [1,2,3,4] =becomes> [2,3,4,1]
        y = y[:-1]  # cut last
        x = x_raw[:-1]  # [1,2,3]

        x_len = len(x)
        pad_right = self.max_sentence_length - x_len
        x_padded = np.pad(x, (0, pad_right))
        y_padded = np.pad(x, (0, pad_right))

        # TODO
        return x_padded, y_padded, x_len


"""
    rt = sigmoid(Wr * xt + Ur * ht-1 + br)
    zt = sigmoid(Wz * xt + Uz * ht-1 + bz)
    h_t_tilde = tanh(Wh * xt + Uh * (rt ⊙ ht-1) + bh)
    ht = (1 - zt) ⊙ ht-1 + zt ⊙ h_t_tilde

"""

class GRU(torch.nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.W_r = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_r = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )
        self.W_z = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_z = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

        self.W_h = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_h = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

        self.V = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=num_embedding
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_len = pad_packed_sequence(x,
                                                batch_first=True)  # True <- makes longest sentence first (safety switch)
        if hidden is None:
            hidden = torch.zeros(x_unpacked.size(0), RNN_HIDDEN_SIZE).to(
                DEVICE)  # why its a bad idea to put into default parameter - its a pointer that points to heap
            # fn used globally will work with defaut value

        # x_unpacked B, Seq, F
        x_seq = x_unpacked.permute(1, 0, 2)
        outs = []
        for x_t in x_seq:  # (B,F)
            # rt = sigmoid(Wr * xt + Ur * ht-1 + br)
            r_t = torch.sigmoid(self.W_r.forward(x_t) + self.U_r.forward(hidden))
            # zt = sigmoid(Wz * xt + Uz * ht-1 + bz)
            z_t = torch.sigmoid(self.W_z.forward(x_t) + self.U_z.forward(hidden))
            # h_t_tilde = tanh(Wh * xt + Uh * (rt ⊙ ht-1) + bh)
            h_t_pri = torch.tanh(self.W_h.forward(x_t) + r_t * self.U_h.forward(hidden))
            # ht = (1 - zt) ⊙ ht-1 + zt ⊙ h_t_tilde
            hidden = (1 - z_t) * hidden + z_t * h_t_pri

            out = self.V.forward(hidden)  # todo normalization to return vocab.size?
            outs.append(out)
        out_seq = torch.stack(outs)

        out_seq = out_seq.permute(1, 0, 2)  # Seq, B, F -> B. Seq, F

        output = pack_padded_sequence(out_seq, x_len, batch_first=True, enforce_sorted=False)

        return output, hidden


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # h_t = tahn(W @ h_<t-1> + U@x_t + b)
        # o_t = V @ h_t + b_o
        # parameter is simple matrix
        self.W = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U = torch.nn.Linear(
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )
        self.V = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=len(dataset_full.vocabulary_keys)
            # torch.FloatTensor(RNN_HIDDEN_SIZE, len(dataset_full.vocabulary_keys))
        )

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_len = pad_packed_sequence(x,
                                                batch_first=True)  # True <- makes longest sentence first (safety switch)
        if hidden is None:
            hidden = torch.zeros(x_unpacked.size(0), RNN_HIDDEN_SIZE).to(
                DEVICE)  # why its a bad idea to put into default parameter - its a pointer that points to heap
            # (diferent) fns used globally will work with (the same) defaut value

        # x_unpacked B, Seq, F
        x_seq = x_unpacked.permute(1, 0, 2)
        outs = []
        for x_t in x_seq:  # (B,F)
            W_dot_x = self.W.forward(hidden)
            U_dot_x = self.U.forward(x_t)
            hidden = torch.tanh(W_dot_x + U_dot_x)
            out = self.V.forward(hidden)
            outs.append(out)

        out_seq = torch.stack(outs)

        out_seq = out_seq.permute(1, 0, 2)  # Seq, B, F -> B. Seq, F

        output = pack_padded_sequence(out_seq, x_len, batch_first=True, enforce_sorted=False)

        return output, hidden


"""
    ft = sigmoid(Wf * xt + Uf * ht-1 + bf) Forget gate
    it = sigmoid(Wi * xt + Ui * ht-1 + bi) Input gate
    c_t_tilde = tanh(Wc * xt + Uc * ht-1 + bc) Candidate cell state
    ct = ft ⊙ c(t-1) + it ⊙ c_t_tilde Cell state
    ot = sigmoid(Wo * xt + Uo * ht-1 + bo) Output gate
    ht = ot ⊙ tanh(ct) Hidden state

    2) Implement weight saving at the lowest test_loss value
    3) Implement a separate script where model weights can be loaded, and a user can input a sentence start with several words into the console, and the model will predict the sentence ending
    4) Implement the built-in torch.nn.LSTM model and compare the results with the self-created model
    5) Implement one or more SOTA (State Of The Art) models for time series processing that do not use the Transformer, such as:
            S5 models https://github.com/i404788/s5-pytorch
            LRU (Linear Recurrent Unit) https://github.com/TingdiRen/LRU_pytorch
            RWKV-8 https://github.com/BlinkDL/RWKV-LM xLSTN https://github.com/NX-AI/xlstm

"""


class LSTM(torch.nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.ct = torch.zeros(EMBEDDING_SIZE,RNN_HIDDEN_SIZE).to(DEVICE)

        self.W_f = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_f = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )
        self.W_i = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_i = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

        self.W_c = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_c = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

        self.W_o = torch.nn.Linear(  # allows to call forwards pass directly
            in_features=EMBEDDING_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE), bias=False
        )
        self.U_o = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=RNN_HIDDEN_SIZE
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

        self.V = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=num_embedding
            # torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE)
        )

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_len = pad_packed_sequence(x,
                                                batch_first=True)  # True <- makes longest sentence first (safety switch)
        if hidden is None:
            hidden = torch.zeros(x_unpacked.size(0), RNN_HIDDEN_SIZE).to(
                DEVICE)  # why its a bad idea to put into default parameter - its a pointer that points to heap
            # fn used globally will work with defaut value

        # x_unpacked B, Seq, F
        x_seq = x_unpacked.permute(1, 0, 2)
        outs = []
        for x_t in x_seq:  # (B,F)

            # ft = sigmoid(Wf * xt + Uf * ht-1 + bf) Forget gate
            f_t = torch.sigmoid(self.W_f.forward(x_t) + self.U_f.forward(hidden)) #32,128

            if self.ct.shape[0] != f_t.shape[0]: # reset cell state between different sized batches
                self.ct = torch.zeros(f_t.shape).to(DEVICE)  # 32,128 -> 8,128
            #    it = sigmoid(Wi * xt + Ui * ht-1 + bi) Input gate
            i_t = torch.sigmoid(self.W_i.forward(x_t) + self.U_i.forward(hidden)) #32,128

            #     c_t_tilde = tanh(Wc * xt + Uc * ht-1 + bc) Candidate cell state
            c_t_tilde = torch.tanh(self.W_c.forward(x_t) + self.U_c.forward(hidden)) #32,128

            #     ct = ft ⊙ c(t-1) + it ⊙ c_t_tilde Cell state
            try:
                ct = f_t * self.ct + i_t * c_t_tilde
                self.ct = ct.detach().to(DEVICE)
            except Exception as e:
                print(e)


            #     ot = sigmoid(Wo * xt + Uo * ht-1 + bo) Output gate
            o_t = torch.sigmoid(self.W_o.forward(x_t) + self.U_o.forward(hidden)) #32,128

            #     ht = ot ⊙ tanh(ct) Hidden state
            hidden = o_t * torch.tanh(self.ct)

            out = self.V.forward(hidden)
            outs.append(out)
        out_seq = torch.stack(outs)

        out_seq = out_seq.permute(1, 0, 2)  # Seq, B, F -> B. Seq, F

        output = pack_padded_sequence(out_seq, x_len, batch_first=True, enforce_sorted=False)

        return output, hidden


class Model(torch.nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.num_embedding = num_embedding
        # embeds
        self.emb = torch.nn.Embedding(
            num_embeddings=num_embedding,
            embedding_dim=EMBEDDING_SIZE
        )

        if USE_GRU:
            self.rnn = GRU(num_embedding)
        elif USE_LSTM:
            self.rnn = LSTM(num_embedding)
        else:
            self.rnn = RNN()

    def forward(self, x: PackedSequence, hidden=None):
        # B,Seq, 1 -> B, Seq, Emb
        x_emb = self.emb.forward(x.data)  # x.dat is the sousage of all sentences (packedsequenceofsentences)
        x_emb_packed = PackedSequence(
            data=x_emb,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        out, hidden = self.rnn.forward(x_emb_packed, hidden)
        y_prim = torch.softmax(out.data, dim=-1)  # sousage probabs of words

        y_prim_packed = PackedSequence(
            data=y_prim,
            # tokens where we need to cut the sousage
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        return y_prim_packed, hidden

def rollout(model: Model, inp_padded:torch.Tensor, max_sentence_len, vocabulary):
    hidden = None

    x_roll = inp_padded[:, :1] # 28,61 -> 28,1
    batch_size = x_roll.size(0)

    # step by step autoregressive
    for t in range(max_sentence_len):
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_roll[:, -1:],
            lengths=torch.LongTensor([1] * batch_size),  # [1, 1, 1, 1 ...]
            batch_first=True,
        ).to(DEVICE)

        y_prim, hidden = model.forward(x_packed, hidden) # y_prim:1,11258;

        y_prim_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(y_prim, batch_first=True)
        y_prim_idx = y_prim_unpacked.argmax(dim=-1)
        x_roll = torch.cat((x_roll, y_prim_idx), dim=-1).to(DEVICE)

    np_x_roll = x_roll.cpu().numpy()
    for sent in np_x_roll:
        words = [vocabulary[it] for it in sent]
        if '[eos]' in words:
            eos_idx = words.index('[eos]')
            words = words[:eos_idx]
        #print(f'{epoch}: ', end='')
        print(' '.join(words))

if __name__ == "__main__":

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
        drop_last=(len(dataset_train) % BATCH_SIZE == 1),
        ## its important to drop last if 1 batch norm will fail because cant count  mean and sigma for one sample
        shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        drop_last=(len(dataset_test) % BATCH_SIZE == 1),
        shuffle=False

    )

    model = Model(len(dataset_full.vocabulary_keys))
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_weights = torch.FloatTensor(
        1. / np.array((dataset_full.vocabulary_counts)))  # inverse a/counts, a - multiplier to increase
    # instead dataset_full, should be dataset_train
    loss_weights = loss_weights.to(DEVICE)

    loss_plot_train = []
    loss_plot_test = []
    acc_plot_train = []
    acc_plot_test = []

    fig, (ax1, ay1) = plt.subplots(2, 1)
    plt.ion()

    best_test_loss = float('Inf')

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
                # L_cce = -y*log(y') = -log(y'_<y_idx>) < doing cross entropy for the specific y_idx
                loss = -torch.mean(loss_weights[idxes_y] * torch.log(y_prim_packed.data[idxes_batch, idxes_y] + 1e-8))
                losses.append(loss.cpu().item())

                idxes_y_prim = y_prim_packed.data.argmax(dim=-1)
                acc = torch.mean((idxes_y_prim == idxes_y) * 1.0)
                accs.append(acc.cpu().item())

                if dataloader == dataloader_train:
                    """
                        Traverses the computation graph backward
                            Starting from the tensor you call .backward() on, PyTorch moves backward through all operations that produced it.
                        Computes gradients (∂output/∂input)
                            It calculates the derivative of that tensor with respect to each leaf tensor (parameters, inputs) that has requires_grad=True.
                        Stores the results in .grad
                            After calling .backward(), all leaf tensors get their gradients stored in the .grad attribute.
                        Copied from: Tensor.backward explanation - <https://chatgpt.com/c/692ada6a-7f8c-832e-b27d-7d57b2101699>
                    """
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            if dataloader == dataloader_train:
                loss_plot_train.append(np.mean(losses))
                acc_plot_train.append(np.mean(accs))
            else:
                loss_plot_test.append(np.mean(losses))
                acc_plot_test.append(np.mean(accs))


        if best_test_loss > round(loss_plot_test[-1], SAVE_STATE_TEST_LOSS_SENSITIVITY):
            print(f' Saving best test_loss {loss_plot_test[-1]} weights...')
            best_test_loss = round(loss_plot_test[-1], SAVE_STATE_TEST_LOSS_SENSITIVITY)
            #todo save: model state (load state), vocabulary (rollout), max sentence len, embed size, hidden size,
            checkpoint = {
                'state': model.state_dict(),
                'vocab': dataset_full.vocabulary_keys,
                'max_sentence_len': dataset_full.max_sentence_length,
                'embed_size': EMBEDDING_SIZE,
                'hidden_size': RNN_HIDDEN_SIZE
            }

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(checkpoint, f'{checkpoint_path}/{ch_file}')


        if epoch % 10 == 0:
            print(
                f'\n\nepoch: {epoch} '
                f'loss_train: {loss_plot_train[-1]:.3f} '
                f'loss_test: {loss_plot_test[-1]:.4f} '
                f'acc_plot_train: {acc_plot_train[-1]:.3f} '
                f'acc_plot_test: {acc_plot_test[-1]:.3f} '
            )

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
            rollout(model, inp_padded=x_padded, max_sentence_len=dataset_full.max_sentence_length, vocabulary=dataset_full.vocabulary_keys)