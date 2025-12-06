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
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10_000.0) / embedding_dim))
        # sinusoidal way of representing data  -- visual way of creating embedding without learning it
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, idxes):
        return self.pe[idxes, :]


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Project inputs to queries, keys and values (Q, K, V).
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
        # Then reshape to (B, Heads, Seq, head_dim) to implement multi-head attention
        # This implements the "Multi-Head Attention" and "Scaled Dot-Product Attention" in the paper.
        k = k.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)
        q = q.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)
        v = v.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)

        # Scaled dot-product: attention logits = Q @ K^T / sqrt(d_k)
        # A causal lower-triangular mask is applied so tokens cannot attend to future tokens (autoregressive).
        # Positions beyond sentence length are also masked (padding).
        atten_raw = q @ k.transpose(-1, -2) / np.sqrt(x.size(-1)) # use head_dim = HIDDEN_SIZE/HEADS
        mask = torch.tril(torch.ones(seq_size, seq_size)).to(DEVICE) # (Seq, Seq) causal mask
                                                                     # cut lower part of matrix - necessary only for prediction tasks;
                                                                     # translation instead sees the whole context
        trick = float('-inf')  # a  trick for softmax to not consider those values at all
        atten_mask = atten_raw.masked_fill(mask == 0, value=trick)  # (B, H, Seq, Seq)

        for idx, length in enumerate(lengths): # (B, Seq, Seq) # ensure that model trains only on the important parts
            atten_mask[idx, :, :, length:] = trick
            atten_mask[idx, :, length:, :] = trick

        atten = torch.softmax(atten_mask, dim=-1)
        atten = atten.masked_fill(((atten > 0) == False), value=0.0)  # another trick to hide zeroes, which comes next
        out = atten @ v

        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_size, HIDDEN_SIZE) # restore split head into one (recombine) will be exactly the same  as single-head
        atten = atten.detach().mean(dim=1) # shape (B, Heads, seq, seq) => (B, seq, seq) # pick mean for visualization

        # torch.nn.Module > self.training
        # model.eval() model.train()
        # Add & Norm: residual connection followed by LayerNorm (post-norm style).
        # Then a position-wise feed-forward network is applied to each position independently.
        # This matches the "Add & Norm" and "Feed-Forward" block of each Transformer layer.
        out_1 = x + torch.dropout(out, p=DROPOUT, train=self.training) # randomly drops some entries from attention matrix,
                                                                       # not to learn identical sequence, force it to adapt to changes
        # final norm steps                                             # when inference we don't do that to not lose the context: P and train are the flags for switching mode
        out_1_norm = self.norm_1.forward(out_1)
        out_2 = self.ff.forward(out_1_norm) # position-wise FFN
        out_3 = out_1_norm + out_2
        y_prim = self.norm_2.forward(out_3)

        return y_prim, lengths, atten


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Token embedding (maps token ids -> continuous d-dimensional vectors).
        # This corresponds to "Word Embeddings" in the Transformer paper.
        # NB these weights are also used as output projection (weight tying) later.
        self.project_w_e = torch.nn.Embedding(
            num_embeddings=len(dataset_full.vocabulary_keys), # the amount model can predict i.e. tokens
            embedding_dim=HIDDEN_SIZE
        )

        # Positional encoding — provides tokens with position information.
        # The original Paper used sinusoidal PositionalEncoding; this code uses learnable positional embeddings (nn.Embedding).
        self.project_p_e = torch.nn.Embedding(
            num_embeddings=dataset_full.max_sentence_length, #hardware dependant
            embedding_dim=HIDDEN_SIZE
        )

        self.transformer = torch.nn.ModuleList( #sequential <- cant debug into; module list -> allows iterating over
            [TransformerLayer() for _ in range(TRANSFORMER_LAYERS)]
        )

        self.fc = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, x: PackedSequence):

        x_e = PackedSequence(
            data=self.project_w_e.forward(x.data),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        x_e_unpacked, lengths = pad_packed_sequence(x_e, batch_first=True) # batch, seq_length, hidd_length
                                                                           # decrease the size of input by stacking, and removing zeroes; ~50% savings

        # 0, 1, 2, 3, 4.... => (B, Seq) => project_p_e => (B, Seq, HIDDEN_SIZE)
        # lengths[0]
        pos_idxes = torch.arange(0, torch.max(lengths)).to(DEVICE) # size of specific batch
        p_e = self.project_p_e.forward(pos_idxes) # (Seq,)
        p_e = p_e.unsqueeze(dim=0) # (1, Seq, H)
        p_e = p_e.expand(x_e_unpacked.size()) # return to initial dimensions

        z = x_e_unpacked + p_e  # x*W_e + W_p
        atten = None
        for layer in self.transformer:
            z, lengths, atten = layer.forward(z, lengths, atten)

        z_packed = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False) # pack to be consistent (todo as it wasnt)
        out_fc = self.fc.forward(z_packed.data)

        # Final projection to vocabulary:
        # - project last hidden vectors to logits
        # - apply softmax to get token probabilities
        # This corresponds to the final linear + softmax in the paper (and uses weight tying).
        y_prim_logits = (self.project_w_e.weight @ out_fc.unsqueeze(dim=-1)).squeeze(dim=-1) # transposed embedding step
        y_prim = torch.softmax(y_prim_logits, dim=-1)

        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        return y_prim_packed, atten # the atten matrix after processing is not used unlike RNNs; here we return just for visualization


model = Model()
#model = model.to(DEVICE)

if not os.path.exists(f'{PATH_DATA}/pretrain-model-259.pt'):
    download_url_to_file('http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/pretrain-model-259.pt', f'{PATH_DATA}/pretrain-model-259.pt', progress=True)

model.load_state_dict(
    state_dict=torch.load(
        f'{PATH_DATA}/pretrain-model-259.pt',
        map_location='cpu'
    )
)
#model = model.train() # in training mode dropout is enabled which causes random neuron disabling
                        # if enabled for inference then predictions will be unstable
model = model.eval() #todo train <- fix multiple stopwords
torch.set_grad_enabled(False)

while True:
    x_str = input('x: (exit with \q)')
    x_list = x_str.lower().split(' ')

    is_valid = True
    x_idxes = []
    for x_each in x_list: # 1. tokenize user input
        if x_each == '\q':
            exit()
        if x_each not in dataset_full.vocabulary_keys:
            print(f"\033[91m {x_each} not found in vocabulary \033[0m")
            is_valid = False
        else:
            x_idxes.append(dataset_full.vocabulary_keys.index(x_each))

    if not is_valid:
        continue

    ## run inference on user input
    # 1. tokenize user input
    # 2. run prediction loop for max_tokens to generate (rollout)
    # 3. transforms ids to words
    cur_len = len(x_idxes)
    max_len = dataset_full.max_sentence_length
    max_tokens = max_len - cur_len
    idx_stop = dataset_full.vocabulary_keys.index('[eos]')

    """
        the rollout process (Rollout = autoregressive generation): 
         - Generating new tokens step-by-step, feeding each predicted token back into the model again.
        It is used for text generation, sequence completion, language modeling, etc.
    """
    atten = None
    for step in range(max_tokens):
        cur_len = len(x_idxes)

        #pad input
        x_padded = np.pad(x_idxes, (0, max_len - cur_len))
        x_tensor = torch.tensor(x_padded).unsqueeze(0)
        x_len_tensor = torch.tensor([cur_len])

        x_packed = pack_padded_sequence(x_tensor, x_len_tensor, batch_first=True, enforce_sorted=False)

        y_prim_packed, atten = model.forward(x_packed)

        idxes_y_prim = y_prim_packed.data.argmax(dim=-1)

        if idxes_y_prim[-1] == idx_stop:
            break

        x_idxes.append(idxes_y_prim[-1])
        x_list.append(dataset_full.vocabulary_keys[idxes_y_prim[-1]])

        print(f' Step {step}: ')
        for i in idxes_y_prim:
            print(f"{dataset_full.vocabulary_keys[i]} ", end='') #todo why some words in the middle are changed?
        print(f'{x_idxes}')

    atten = atten[0] # remove batch -> (seq, seq)

    """
        Implement and train model using torch.nn.TransformerEncoderLayer
    """
    print('>>> Output >>>>>>>>>>>>>>>>>>>>>')
    print('y_prim: ' + ' '.join(x_list))
    print(f'{atten}')

    plt.imshow(atten, aspect='auto', colorizer='gray')
    plt.title('Attention matrix')
    plt.xlabel('Attention')
    plt.ylabel('Words')
    plt.xticks(range(len(x_idxes)), x_list, rotation=35)
    plt.yticks(range(len(x_idxes)), x_list)
    plt.colorbar()
    plt.show()

