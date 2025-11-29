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

from prd10.rnn_live11_2 import Model, rollout, USE_LSTM,USE_GRU

checkpoint_path = 'data/rnn_chpoints'

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

"""
    Implement a separate script where model weights can be loaded, 
    and a user can input a sentence start with several words into the console, 
    and the model will predict the sentence ending

    {state, vocab, max_sentence_len, embed_size, hidden_size}
"""
ch_file = 'rnn_best.pt'
if USE_LSTM:
    ch_file = 'lstm_best.pt'
if USE_GRU:
    ch_file = 'gru_best_local.pt'
checkpoint = torch.load(f'{checkpoint_path}/{ch_file}', weights_only=False) # use weights_only=False when stored a custom object

vocabulary = checkpoint['vocab']
vocWord2Idx = {word:idx for idx,word in enumerate(vocabulary)} # transform list to dictionary

num_embedings = len(vocabulary)
EMBEDDING_SIZE = checkpoint['embed_size']
RNN_HIDDEN_SIZE = checkpoint['hidden_size']
max_sentence_len = checkpoint['max_sentence_len']

model = Model(num_embedings)
load_result = model.load_state_dict(checkpoint['state']) #.to(DEVICE)

print(f' Model loaded {load_result}')
model.to(DEVICE)
model.eval()

IDX_UNKNOWN_TOKEN = 1 # replace with special token id
inp = "they only where" #input('Input sentence start for inference:')
#inp = "weak"
inp_tokens = inp.split()
inp_idxs = [vocWord2Idx.get(token, IDX_UNKNOWN_TOKEN) for token in inp_tokens] # todo check that token exist and return , IDX_UNKNOWN_TOKEN) if not
inp_padded = inp_idxs + [0]*(max_sentence_len - len(inp_tokens)) # use np.pad(x, pad_width)
inp_tensor = torch.tensor([inp_padded]).to(DEVICE)

rollout(model, inp_tensor, max_sentence_len, vocabulary)
if False:
    inp_tensor_lengths = [1]*inp_tensor.size(-1)

    inp_packed = torch.nn.utils.rnn.pack_padded_sequence(inp_tensor,
                                                         lengths=inp_tensor_lengths  #  list of sequence lengths of each batch element (must be on the CPU if provided as a tensor).
                                                         ).to(DEVICE)

    y_prim_packed, _ = model.forward(inp_packed)
    y_prim_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(y_prim_packed, batch_first=True)
    idxes_y_prim = y_prim_unpacked.data.argmax(dim=-1)

