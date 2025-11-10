import os
import pickle
import random

import torch.nn.functional as F

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
from torch.hub import download_url_to_file
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
plt.style.use('dark_background')


import torch.utils.data
import scipy.misc
import scipy.ndimage


USE_CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 64
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if USE_CUDA:
    MAX_LEN = None

class DatasetApples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/apples_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/apples_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        X = torch.from_numpy(np.array(X))
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = F.one_hot(Y)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y = self.Y[idx]

        return x, y


dataset_full = DatasetApples()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
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

class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 =  torch.nn.BatchNorm2d(out_channels)
        self.conv2 =  torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 =  torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1)

    def forward(self, x):
        # TODO
        out = self.conv1.forward(x)
        out = torch.relu(out)
        out = self.bn1.forward(out)
        out = self.conv2.forward(out)

        residual = self.shortcut.forward(out)

        out += residual
        out = torch.relu(out)
        out = self.bn2.forward(out)

        return out

class DenseBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)

        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.bn1 = torch.nn.BatchNorm2d(num_features=64) #stacking on top 32+32
        self.bn1 = torch.nn.BatchNorm2d(num_features=96) # 64+32
        self.bn1 = torch.nn.BatchNorm2d(num_features=128) # 96+32

    def forward(self, x):
        conv1 = self.conv1.forward(x)
        conv1 = torch.relu(conv1)
        conv1 = self.bn1.forward(conv1)

        conv2_in = torch.cat([x, conv1], dim=1)
        conv2 = self.conv2.forward(conv2_in)
        conv2 = torch.relu(conv2)
        conv2 = self.bn1.forward(conv2)

        conv3_in = torch.cat([x, conv1, conv2], dim=1)
        conv3 = self.conv3.forward(conv3_in)
        conv3 = torch.relu(conv3)
        conv3 = self.bn1.forward(conv3)

        conv4_in = torch.cat([x, conv1, conv2, conv3], dim=1)
        conv4 = self.conv4.forward(conv4_in)
        conv4 = torch.relu(conv4)
        conv4 = self.bn1.forward(conv4)

        out = torch.cat([x, conv1, conv2, conv3, conv4], dim=1)

        # TODO
        return x

# to adapt changes in the sizes
class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv.forward(x)
        out = torch.relu(out)
        out = self.bn.forward(out)

        out = self.avg_pool(out)

        return out

class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.densebl1 = DenseBlock()
        self.densebl2 = DenseBlock()
        self.densebl3 = DenseBlock()

        self.trans_layer01 = TransitionLayer(in_channels=160, out_channels=32)
        self.trans_layer02 = TransitionLayer(in_channels=160, out_channels=32)
        self.trans_layer03 = TransitionLayer(in_channels=160, out_channels=32)

        self.linear = torch.nn.Linear(in_features=32*3*3, out_features=len(dataset_full.labels))

    def forward(self, x):
        out = self.conv1.forward(x)

        out = self.densebl1.forward(out)
        out = self.trans_layer01.forward(out)
        out = self.densebl2.forward(out)
        out = self.trans_layer02.forward(out)
        out = self.densebl3.forward(out)
        out = self.trans_layer03.forward(out)

        return x


class ModelResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=1,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=8)

        self.identity_bl1 = ResBlock(in_channels=8, out_channels=8)
        self.identity_bl2 = ResBlock(in_channels=8, out_channels=8)

        # changing the size
        self.projection_bl1 =  ResBlock(in_channels=8, out_channels=16, stride=2)
        self.identity_bl3 = ResBlock(in_channels=16, out_channels=16)

        self.projection_bl2 =  ResBlock(in_channels=16, out_channels=32, stride=2)
        self.identity_bl4 = ResBlock(in_channels=32, out_channels=32)

        self.projection_bl3 =  ResBlock(in_channels=32, out_channels=64, stride=2)
        self.identity_bl5 = ResBlock(in_channels=64, out_channels=64)

        self.linear = torch.nn.Linear(in_features=64, out_features=len(dataset_full.labels))

    def forward(self, x):
        out = self.conv1.forward(x)
        out = torch.relu(out)
        out = self.bn1.forward(out)

        out = self.identity_bl1(out)
        out = self.identity_bl2(out)

        out = self.projection_bl1(out)
        out = self.identity_bl3(out)
        out = self.projection_bl2(out)
        out = self.identity_bl4(out)
        out = self.projection_bl3(out)
        out = self.identity_bl5(out)

        out = F.avg_pool2d(out, 3) # will reduce size further

        #compress last 3 dim into single
        out = out.view(out.size(0), -1)

        out = self.linear.forward(out)
        #out = F.softmax() <- only one may be the winner or sigmoid -> [0..1] all of the classes maybe at the same time
        out = F.softmax(out, dim=1)

        # TODO
        return out

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=5)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1))
        out = out.view(x.size(0), -1)
        out = F.softmax(out, dim=1)
        return out

model = Model()
loss_func = LossCrossEntropy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

if USE_CUDA:
    model = model.to('cuda:0')
    # model = model.cuda()
    loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in tqdm(data_loader):
            # #CAUTION random resize here!!! model must work regardless
            # out_size = int(28 * (random.random() * 0.3 + 1.0))
            # x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(out_size, out_size))

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_prim = model.forward(x)
            loss = loss_func.forward(y, y_prim)
            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.cpu()
            y_prim = y_prim.cpu()
            x = x.cpu()
            y = y.cpu()

            np_y_prim = y_prim.data.numpy()
            np_y = y.data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(121) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 10, 11, 12, 16, 17, 18]):
        plt.subplot(3, 6, j)
        color = 'green' if idx_y_prim[i]==idx_y[i] else 'red'
        plt.title(f"pred: {idx_y_prim[i]}\n real: {idx_y[i]}", c=color)
        plt.imshow(x[i].permute(1, 2, 0))

    plt.tight_layout(pad=0.5)
    plt.draw()
    plt.pause(0.1)

input('quit?')
