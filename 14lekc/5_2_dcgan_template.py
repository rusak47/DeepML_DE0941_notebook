import argparse # pip3 install argparse
from copy import copy

from tqdm import tqdm # pip install tqdm
import hashlib
import os
import pickle
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import random
import torch.distributed
import torch.multiprocessing as mp

import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-classes_count', default=3, type=int)
parser.add_argument('-samples_per_class', default=10000, type=int)

parser.add_argument('-learning_rate', default=3e-4, type=float)
parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-is_debug', default=True, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count # 0 = include all
IS_DEBUG = args.is_debug
INPUT_SIZE = 28

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 300 # per class for debugging
    MAX_CLASSES = 6 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 66

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='balanced',
            train=(MAX_LEN == 0),
            download=True
        )
        class_to_idx = self.data.class_to_idx
        idx_to_class = dict((value, key) for key, value in class_to_idx.items())
        self.labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.transpose(np.array(pil_x)).astype(np.float32)
        np_x = np.expand_dims(np_x, axis=0) / 255.0 # (1, W, H) => (1, 28, 28)
        return np_x, y_idx


class ModelD(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #(B, 1, 28, 28) -> (B, 1) True or False
        self.encoder = torch.nn.Sequential(# recommended sequence
            torch.nn.Conv2d(in_channels=1, out_channels=8,kernel_size=7, stride=1, padding=1), # dont change the size (local linear layer
            torch.nn.InstanceNorm2d(num_features=8), # normalize just 1 image - most simplest norm
            torch.nn.GELU(), # (non linearity)
            torch.nn.AdaptiveMaxPool2d(output_size=(14,14)),

            torch.nn.Conv2d(in_channels=8, out_channels=16,kernel_size=5, stride=1, padding=1),            # dont change the size (local linear layer
            torch.nn.InstanceNorm2d(num_features=16),  # normalize just 1 image - most simplest norm
            torch.nn.GELU(),  # (non linearity)
            torch.nn.AdaptiveMaxPool2d(output_size=(7, 7)),

            torch.nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3, stride=1, padding=1),            # dont change the size (local linear layer
            torch.nn.InstanceNorm2d(num_features=32),  # normalize just 1 image - most simplest norm
            torch.nn.GELU(),  # (non linearity)
            torch.nn.AdaptiveMaxPool2d(output_size=(1,1)), # automatically calculates
        #)
            torch.nn.Flatten(),
        # cant add directly to encoder because -> dimension mismatch - There should be Flatten layer to remove redundant dimension
        #self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=1), # score how good image is
           # torch.nn.Sigmoid()
        )
        # softmax if there are multiple outputs

    def forward(self, x):
        y_prim = self.encoder.forward(x)
        return y_prim


class ModelG(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.proj = torch.nn.Linear(Z_SIZE, 16*(INPUT_SIZE//4)**2) # (B, 16, 8, 8) square for width and height
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.InstanceNorm2d(num_features=8),
            torch.nn.GELU(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2) ,# (B, 8, 16, 16) #maybeused AdaptiveMaxPool2d

            torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.InstanceNorm2d(num_features=4),
            torch.nn.GELU(),

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.InstanceNorm2d(num_features=4),
            torch.nn.GELU(),
            torch.nn.UpsamplingBilinear2d(size=(28,28)),  # (B, 8, 16, 16) #maybeused AdaptiveMaxPool2d

            torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, z):
        # (B, 1, 28, 28)
        out = self.proj.forward(z)
        out = out.view(-1, 16,7,7)
        y_prim = self.decoder.forward(out)
        return y_prim


dataset_full = DatasetEMNIST()

#np.random.seed(2)
random.seed(2)
labels_train = copy(dataset_full.labels)
random.shuffle(labels_train)
labels_train = labels_train[:MAX_CLASSES]
np.random.seed(int(time.time()))

idx_train = []
str_args_for_hasing = [str(it) for it in [MAX_LEN, MAX_CLASSES] + labels_train]
hash_args = hashlib.md5((''.join(str_args_for_hasing)).encode()).hexdigest()
path_cache = f'../data/{hash_args}_gan.pkl'
if os.path.exists(path_cache):
    print('loading from cache')
    with open(path_cache, 'rb') as fp:
        idx_train = pickle.load(fp)

else:
    labels_count = dict((key, 0) for key in dataset_full.labels)
    for idx, (x, y_idx) in tqdm(
            enumerate(dataset_full),
            'splitting dataset',
            total=len(dataset_full)
    ):
        label = dataset_full.labels[y_idx]
        if MAX_LEN > 0:
            if labels_count[label] >= MAX_LEN:
                if all(it >= MAX_LEN for it in labels_count.values()):
                    break
                continue
        labels_count[label] += 1
        if label in labels_train:
            idx_train.append(idx)

    with open(path_cache, 'wb') as fp:
        pickle.dump(idx_train, fp)

dataset_train = torch.utils.data.Subset(dataset_full, idx_train)
data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE < 12),
    num_workers=(8 if not IS_DEBUG else 0)
)

model_D = ModelD().to(DEVICE)
model_G = ModelG().to(DEVICE)

optimizer_D = torch.optim.RAdam(model_D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_G = torch.optim.RAdam(model_G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
#optimizer = torch.optim.Adam(list(model_D.parameters()) + list(model_G.parameters()), lr=LEARNING_RATE)

metrics = {}
for stage in ['train']:
    for metric in ['loss', 'loss_g', 'loss_d']:
        metrics[f'{stage}_{metric}'] = []

distribution = torch.distributions.Normal(loc=0., scale=1.0) # mean - loc,


for epoch in range(1, EPOCHS+1):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    for x, x_idx in tqdm(data_loader_train, desc=stage):
        x = x.to(DEVICE)

        loss_D = torch.FloatTensor([0])
        loss_G = torch.FloatTensor([0])

        y_real = model_D.forward(x) # training descriminator
        z = distribution.sample(sample_shape=(x.size(0), Z_SIZE)).to(DEVICE)

        x_fake = model_G.forward(z)
        model_G.eval()
        model_G.requires_grad_(False) # disable training at this moment
        x_fake =  model_G.forward(z)
        y_fake = model_D.forward(x_fake.detach())

        loss_D = -torch.mean(y_real - y_fake)
        loss_D.backward()

        torch.nn.utils.clip_grad_norm_(model_D.parameters(), max_norm=1e-2, norm_type=1)
        optimizer_D.step()
        optimizer_D.zero_grad()

        model_G.train()
        model_G.requires_grad_(True)

        # training generator
        z = distribution.sample(sample_shape=(x.size(0), Z_SIZE)).to(DEVICE)
        x_fake = model_G.forward(z)

        model_D.eval()
        model_D.requires_grad_(False)
        y_fake = model_D.forward(x_fake)

        loss_G = -torch.mean(y_fake) # minus means - make it larger
        loss_G.backward()

        torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1e-2, norm_type=1)
        optimizer_G.step()
        optimizer_G.zero_grad()

        model_D.train()
        model_D.requires_grad_(True)

        loss = loss_D + loss_G
        #loss.backward()

        # we can use one optimizer, but 2 is better
        #optimizer.step()
        #optimizer.zero_grad()

        #metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
        metrics_epoch[f'{stage}_loss_g'].append(loss_G.cpu().item())
        metrics_epoch[f'{stage}_loss_d'].append(loss_D.cpu().item())

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(121) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    plt.subplot(122)  # row col idx
    grid_img = torchvision.utils.make_grid(
        x_fake.detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.tight_layout(pad=0.5)

    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss'][-1]) or np.isinf(metrics[f'train_loss'][-1]):
            exit()
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')


