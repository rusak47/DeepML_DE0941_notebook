import os
import pickle
import time
from collections import Counter

import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import sklearn.model_selection

plt.rcParams["figure.figsize"] = (12, 14) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-1
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8
EMBEDDING_SIZE = 8

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset_cce.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/cardekho_india_dataset_cce.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        X = np.array(X)
        self.Y_idx = Y
        self.Y_labels = self.labels[3]
        self.Y_len = len(self.Y_labels)

        Y_counter = Counter(Y)
        Y_counter_val = np.array(list(Y_counter.values()))
        self.Y_weights = (1.0 / Y_counter_val) * np.sum(Y_counter_val)
        print(f'self.Y_weights: {self.Y_weights}')

        self.X_classes = np.array(X[:, :3])

        self.X = np.array(X[:, 3:]).astype(np.float32)  # VERY IMPORTANT OTHERWISE NOT ENOUGH CAPACITY
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std
        self.X = self.X.astype(np.float32)

        # x_brands,
        # x_transmission,
        # x_seller_type,

        # x_year,
        # x_km_driven,
        # y_owner,
        # y_selling_price

        self.Y = np.zeros((len(Y), self.Y_len)).astype(np.float32)
        self.Y[range(len(Y)), Y] = 1.0

        # x_fuel
        # Diesel
        # Petrol
        # CNG
        # LPG

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], np.array(self.X_classes[idx]), self.Y[idx]


class DataLoader:
    def __init__(
            self,
            dataset,
            idxes,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idxes = np.array(idxes)
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return len(self.idxes) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch > len(self):
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size
        idx_end = idx_start + self.batch_size
        idxes_batch = self.idxes[idx_start:idx_end]
        if len(idxes_batch) < BATCH_SIZE:
            raise StopIteration()

        batch = self.dataset[idxes_batch]
        self.idx_batch += 1
        return batch

dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

idxes_train, idxes_test = sklearn.model_selection.train_test_split(
    np.arange(len(dataset_full)),
    train_size=train_test_split,
    test_size=len(dataset_full) - train_test_split,
    stratify=dataset_full.Y_idx,
    random_state=0
)

dataloader_train = DataLoader(
    dataset_full,
    idxes=idxes_train,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idxes=idxes_test,
    batch_size=BATCH_SIZE
)

class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W:  Variable = Variable(
            value=np.random.uniform(low=-1, size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b: Variable = Variable(
            value=np.zeros(shape=(out_features,)),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.squeeze(self.W.value.T @ np.expand_dims(x.value, axis=-1), axis=-1) + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += 1 * self.output.grad
        self.W.grad += np.expand_dims(self.x.value, axis=-1) @ np.expand_dims(self.output.grad, axis=-2)
        self.x.grad += np.squeeze(self.W.value @ np.expand_dims(self.output.grad, axis=-1), axis=-1)

# make -inf .. inf into 0-1 range
class LayerSigmoid():
    def __init__(self):
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(1.0 / (1.0 + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1.0 - self.output.value) * self.output.grad

"""
 O/P layer -> softmax(x) -> [probabilities] 
"""
class LayerSoftmax():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        sum = np.sum(x.value)  + 1e-8
        self.output = Variable(
            np.exp(x.value)/sum
        )
        return self.output

    """
        For SoftMax, the partial derivatives turn out to be:
        da_i/dx_j = a_i(1-a_i) if i=j else -a_ia_j

        Copied from: ChatGPT - <https://chatgpt.com/>
    """
    def backward(self):
        #N = self.x.value.shape[0]
        self.x.grad = self.output.value * (self.output.grad - np.sum(self.output.grad * self.output.value, axis=1, keepdims=True))

        #self.x.grad += (self.output.value - y)/N # applicable if softmax combined with CCE into one layer

class LossCrossEntropy():
    def __init__(self):
        self.y = None
        self.y_prim = None
        self.output = None

    # L_cce(y,y') = 1/N sum( -y * log(y' + eps) )
    def forward(self, y:Variable, y_prim:Variable):
        self.y = y
        self.y_prim = y_prim
        self.output = np.mean( -y.value * np.log (y_prim.value + 1e-8))

        return self.output

    # L_cce(y,y')/dy' = 1/N sum( -y / (y' + eps) )
    def backward(self):
        N = self.y.value.shape[0]
        #self.y_prim.grad = np.mean( -self.y.value / (self.y_prim.value + 1e-8))
        self.y_prim.grad = (-self.y.value / (self.y_prim.value + 1e-8))/N   # np.mean() computes the average over all elements in the tensor
                                                                            # — that is, it collapses everything (including the class dimension) into a single scalar.
                                                                            #  But in backpropagation, we need the per-element gradient — the same shape as y_prim.
                                                                            #  Thus keep all element-wise gradients, then normalize by batch size N.
                                                                            # Both methods  are equivalent when computed per element,
                                                                            # but in code np.mean() would collapse your gradient vector entirely, which is wrong for backprop.   (c) chatgpt

class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.x_indexes = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_m = Variable(np.random.random((num_embeddings, embedding_dim)))
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x_indexes = x.value.squeeze().astype(np.int32)
        self.output = Variable(np.array(self.emb_m.value[self.x_indexes, :])) # same as dot product with one-hot encoded X and Emb_w
        return self.output

    def backward(self):
        self.emb_m.grad[self.x_indexes, :] += self.output.grad


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=4 + EMBEDDING_SIZE * 3, out_features=40),
            LayerSigmoid(),
            LayerLinear(in_features=40, out_features=40),
            LayerSigmoid(),
            LayerLinear(in_features=40, out_features=4),
            LayerSoftmax()
        ]

        self.x_concat = None
        self.embs = []
        for i in range(3):
            self.embs.append(
                LayerEmbedding(embedding_dim=EMBEDDING_SIZE, num_embeddings=len(dataset_full.labels[i]))
            )

    def forward(self, x, x_classes):

        x_enc = []
        for i in range(len(self.embs)):
            x_enc.append(self.embs[i].forward(Variable(x_classes.value[:, i])).value)
        x_enc = np.concatenate(x_enc, axis=-1)
        out = np.concatenate([x.value, x_enc], axis=-1)
        self.x_concat = Variable(out)

        out = self.x_concat
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

        for idx_emb, x_emb in enumerate(self.embs):
            self.embs[idx_emb].grad = self.x_concat.grad[:, idx_emb*EMBEDDING_SIZE:(idx_emb+1)*EMBEDDING_SIZE]
            self.embs[idx_emb].backward()

    def parameters(self):
        variables = []
        for emb in self.embs:
            variables.append(emb.emb_m)
        for layer in self.layers:
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # W'= W - dW * alpha
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossCrossEntropy()


loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []

for epoch in range(1, 100):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        nrmse = []
        accs = []
        for x, x_classes, y in dataloader:

            y_prim = model.forward(Variable(value=x), Variable(value=x_classes))
            loss = loss_fn.forward(Variable(value=y), y_prim)

            y_idx = np.argmax(y, axis=-1)
            y_prim_idx = np.argmax(y_prim.value, axis=-1)
            acc = np.mean((y_idx == y_prim_idx) * 1.0)

            losses.append(loss)
            accs.append(acc)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))

    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_train: {acc_plot_train[-1]} '
        f'acc_test: {acc_plot_test[-1]}'
    )

    if epoch % 10 == 0:
        _, axes = plt.subplots(nrows=2, ncols=1)
        ax1 = axes[0]
        ax1.set_title("Loss")
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[1]
        ax1.set_title("Acc")
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Acc.")
        plt.show()