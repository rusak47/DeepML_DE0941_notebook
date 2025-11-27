import json
import os
import pdb
import pickle

import imageio #pip3 install imageio
import torch
import numpy as np
from torch.onnx.ops import attention
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_name', default='', type=str)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-epochs', default=1000, type=int)
parser.add_argument('-learning_rate', default=1e-3, type=float)

parser.add_argument('-hidden_size', default=64, type=int)
parser.add_argument('-max_len', default=0, type=int)

args, args_other = parser.parse_known_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

HIDDEN_SIZE = args.hidden_size
TRANSFORMER_LAYERS = 8
DROPOUT = 0.1
run_name = args.run_name

TRANSFORMER_HEADS = 4

DEVICE = 'cpu'
if False and torch.cuda.is_available():
    DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = args.max_len

PATH_DATA = '../data'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)

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
train_test_split = int(len(dataset_full) * 0.8)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1).to(DEVICE)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10_000)/embedding_dim)).to(DEVICE)
        # sinusoidal way of representing data  -- visual way of creating embedding without learning it
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        #debug into-> plt.imgshow(pe.detach().cpu().numpy()); plt.show()

    def forward(self, idxes):
        return self.pe[idxes, :]


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.project_k = torch.nn.Linear(  in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE )
        self.project_q = torch.nn.Linear(  in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE )
        self.project_v = torch.nn.Linear(  in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE )

        self.ff = torch.nn.Sequential(
            torch.nn.Linear( in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE ),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        )

        self.norm1 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)

    def forward(self, x, lengths, atten):
        batch_size = x.size(0) # x.shape (B, seq, HIDDEN_SIZE)
        seq_size = x.size(1)

        k = self.project_k.forward(x)
        q = self.project_q.forward(x)
        v = self.project_v.forward(x)

        # split the hidden size: add extta dim
        k = k.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS))
        q = q.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS))
        v = v.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS))

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        att_raw = q @ k.transpose(-1,-2) / np.sqrt(x.size(-1))

        mask = torch.tril(torch.ones(seq_size, seq_size)).to(DEVICE) # cut lower part of matrix - necessary only for prediction tasks;
                                                                    # translation sees the whole context to

        trick = float('-inf') # a  trick for softmax to not consider those values at all
        attention_mask = att_raw.masked_fill(mask==0, value=trick) # replace small e ; why? (B, seq, seq)

        for idx, length in enumerate(lengths): # ensure that model trains only on the important parts
            attention_mask[idx, :, length:] = trick
            attention_mask[idx, length:, :] = trick

        atten =  torch.softmax(attention_mask, dim=-1)

        atten = atten.masked_fill((atten > 0) == False, value=0.0) # another trick to hide zeroes, which comes next
        out = atten @ v

        put = out.transpose(1,2)
        out = out.contiguous().view(batch_size, seq_size, HIDDEN_SIZE) # restore splitted head into one (recombine) will be exactly the same  as single-head

        atten, _ = torch.max(atten, dim=-1) # take max over head dim; pick one for visualization

        out_1 = x+torch.dropout(out, p=DROPOUT, train=self.training) # randomly drops some entries from attention matrix,
                                                                    # not to learn identical sequence, force it  to adapt to changes
        #final norm steps                                           # when inference we dont do that to not lose the context: P and train are the flags for switching mode
        out_1_norm = self.norm1.forward(out_1)
        out_2 = self.ff.forward(out_1_norm)
        out_3 = out_1_norm + out_2
        y_prim = self.norm2.forward(out_3)

        return y_prim, lengths, atten


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.project_w_e = torch.nn.Embedding(
            num_embeddings=len(dataset_full.vocabulary_keys), # the amount model can predict i.e. tokens
            embedding_dim=HIDDEN_SIZE # hidden size
        )

        self.project_p_e = PositionalEncoding(
            num_embeddings=dataset_full.max_sentence_length, #hardware dependant
            embedding_dim=HIDDEN_SIZE
        )

        self.transformer = torch.nn.ModuleList( #sequential - cant debug into; module list allows iterating over
            [
                TransformerLayer() for _ in range(TRANSFORMER_LAYERS)
            ]
        )

        self.fc = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, x: PackedSequence):

        #TODO
        x_e = PackedSequence(
            data = self.project_w_e.forward(x.data),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
        )
        x_e_unpacked, lengths = pad_packed_sequence(x_e, batch_first=True) # batch, seq_length, hidd_length
                                                                            # decrease the size of input by stacking, and removing zeroes; ~50% savings

        pos_idx = torch.arange(0, torch.max(lengths)).to(DEVICE) # size of specific batch
        p_e = self.project_p_e.forward(pos_idx)
        p_e = p_e.unsqueeze(dim=0)
        p_e = p_e.expand(x_e_unpacked.size()) # return to initial dimensions

        z= x_e_unpacked+ p_e # x*W_e + W_p

        atten = None
        for layer in self.transformer:
            z, lengths, atten = layer.forward(z, lengths, atten)

        z_packed  = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False) # pack to be consisten as it wasnt
        out_fc =  self.fc.forward(z_packed.data)

        y_prim_logits = (self.project_w_e.weight @ out_fc.unsqueeze(dim=-1)).squeeze(dim=-1) # transposed embedding step
        y_prim= torch.softmax(y_prim_logits, dim=-1)
        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        return y_prim_packed, atten # the atten matrix after processing is not used unlike RNNs; here we return just for visualization

model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

loss_weights = torch.FloatTensor(1 - dataset_full.vocabulary_counts / np.sum(dataset_full.vocabulary_counts))
loss_weights = loss_weights.to(DEVICE)

for epoch in range(1, EPOCHS+1):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y, length in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            x_packed = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
            y_packed = pack_padded_sequence(y, length, batch_first=True, enforce_sorted=False)

            y_prim_packed, atten = model.forward(x_packed)

            idxes_batch = range(len(y_packed.data))
            idxes_y = y_packed.data
            loss = -torch.mean(
                loss_weights[idxes_y] * torch.log(y_prim_packed.data[idxes_batch, idxes_y] + 1e-8)
            )

            idxes_y_prim = y_prim_packed.data.argmax(dim=-1)
            acc = torch.mean((idxes_y_prim == idxes_y) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc.item())

            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 4)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/{run_name}-model-{epoch}.pt')
        model = model.to(DEVICE)

        # pip install imageio
        imageio.imwrite( f'./results/{run_name}-epoch-{epoch}-atten-0.png', (atten[0].detach().cpu().numpy() * 255).astype(np.uint8))

        imageio.imwrite( f'./results/{run_name}-epoch-{epoch}-atten-l.png', (atten[-1].detach().cpu().numpy() * 255).astype(np.uint8))


    print('Examples:')
    y_prim_unpacked, lengths_unpacked = pad_packed_sequence(y_prim_packed.cpu(), batch_first=True)
    y_prim_unpacked = y_prim_unpacked[:5] # 5 examples
    for idx, each in enumerate(y_prim_unpacked):
        length = lengths_unpacked[idx]

        y_prim_idxes = np.argmax(each[:length].data.numpy(), axis=1).tolist()
        x_idxes = x[idx, :length].cpu().data.numpy().tolist()
        y_prim_idxes = [x_idxes[0]] + y_prim_idxes
        print('x     : ' +' '.join([dataset_full.vocabulary_keys[it] for it in x_idxes]))
        print('y_prim: ' +' '.join([dataset_full.vocabulary_keys[it] for it in y_prim_idxes]))
        print('')

    plt.figure(figsize=(12,5))
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig(f'./results/{run_name}-epoch-{epoch}.png')
    plt.show()