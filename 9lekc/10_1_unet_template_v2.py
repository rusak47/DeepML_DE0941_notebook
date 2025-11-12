from PIL import Image
import torch
import numpy as np
import matplotlib
import torchvision
import math
import os
import zipfile
from torch.hub import download_url_to_file
from tqdm import tqdm  # pip3 install tqdm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 15)
plt.style.use('dark_background')
import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition
import argparse  # pip3 install argparse
import torch.nn.functional as F
import torchvision.transforms as transforms

USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = 8
LR = 1e-3


class DatasetBalloons(torch.utils.data.Dataset):
    def __init__(self, is_train):
        curdir = os.getcwd()
        root = 'data'  ## why this bug hapened!?
        if not curdir.endswith('PyCharmMiscProject'):
            rootpath = '../' + root
        path_dataset = f'{root}/baloon'
        if not os.path.exists(path_dataset):
            download_url_to_file(
                f'https://share.yellowrobot.xyz/quick/2025-11-7-16F4B86A-18DA-4064-8EA7-949E919B1CD0.zip',
                f'{path_dataset}.zip',
                progress=True
            )
            zipfile.ZipFile(f'{path_dataset}.zip').extractall(f'./{path_dataset}')
        print(f'{root}; {curdir}; {path_dataset}')
        train_path = path_dataset+'/balloon/train'
        test_path = path_dataset+'/balloon/val'
        print(f'{path_dataset} exist: {os.path.exists(path_dataset)}')
        print(f'{train_path} exist: {os.path.exists(train_path)}')

        self.path_samples = f'{train_path}' if is_train else f'{test_path}'
        self.files_samples = [f for f in os.listdir(self.path_samples) if f.endswith('.jpg')]
        self.count_samples = len(self.files_samples)

        self.transform_x = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.count_samples

    def __getitem__(self, idx):
        x = Image.open(f'{self.path_samples}/{self.files_samples[idx]}')
        y = Image.open(f'{self.path_samples}/{self.files_samples[idx].replace(".jpg", "_mask.png")}')
        x = self.transform_x(x)
        y = self.transform_mask(y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetBalloons(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetBalloons(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=0
)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels / 2))
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels / 2))
        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.gn1(out)
        out = self.conv2(out)
        if self.is_bottleneck:
            residual = self.shortcut(x)
        out = out + residual # cannot be inplace addition
        out = F.relu(out)
        out = self.gn2(out)
        return out


class UNetConcat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: implement UNet with concatenation skip connections

    def forward(self, x):
        # TODO: implement UNet with concatenation skip connections
        return x


class UNetAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: implement UNet with addition skip connections

    def forward(self, x):
        #TODO: implement UNet with addition skip connections
        return torch.sigmoid(out)


def dice_coeficient(predict, target):
    return 0


def bce_loss(predict, target, weight=1):
    return 0


def combined_loss(predict, target):
    return 0


def iou_coeficient(predict, target):
    return 0


model = UNetConcat()
optimizer = torch.optim.RAdam(model.parameters(), lr=LR)

if USE_CUDA:
    model = model.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'dice_coef',
        'IoU'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}
        for x, y in tqdm(data_loader):

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'
                model = model.eval()
                torch.set_grad_enabled(False)
            else:
                model = model.train()
                torch.set_grad_enabled(True)

            y_prim = model.forward(x)
            loss_bce = bce_loss(y_prim, y)
            dice_coef = dice_coeficient(y_prim, y)
            iou_coef = iou_coeficient(y_prim, y)

            # TODO
            loss = 0
            metrics_epoch[f'{stage}_loss'].append(loss.item())
            metrics_epoch[f'{stage}_dice_coef'].append(dice_coef.item())
            metrics_epoch[f'{stage}_IoU'].append(iou_coeficient(y_prim, y).item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.cpu()
            y_prim = y_prim.cpu()
            x = x.cpu()
            y = y.cpu()

            np_x = x.data.numpy()
            np_y_prim = y_prim.data.numpy()
            np_y = y.data.numpy()

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.figure(figsize=(20, 10))
    plt.subplot(121)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        if len(value) > 0 and not np.all(np.isnan(value)):
            plts += plt.plot(value, f'C{c}', label=key)
            c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18]):
        plt.subplot(4, 6, j)
        plt.title('y')
        plt.imshow(np_x[i].transpose(1, 2, 0), interpolation=None)
        plt.imshow(np_y[i][0], cmap='Reds', alpha=0.5, interpolation=None)
        plt.subplot(4, 6, j + 6)
        plt.title('y_prim')
        plt.imshow(np_x[i].transpose(1, 2, 0), interpolation=None)
        plt.imshow(np.where(np_y_prim[i][0] > 0.8, np_y_prim[i][0], 0), cmap='Reds', alpha=0.5, interpolation=None)

    plt.tight_layout(pad=0.5)
    plt.show()

