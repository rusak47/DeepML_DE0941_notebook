import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

USE_MAE = False
#USE_MAE = True
USE_HUBER = True
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7
EMBEDD_DIM=3
HUBER_THRESHOLD = 0.99 # 1 - full MAE; 0 - full MSE

#derivative - a change (const = 0 as it doesnt change)
#gradient - direction of change (point into direction when loss minimizes)

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)

        self.X = np.array(self.X).astype(np.float32)
        # normalized_x, _, _ = self.normalize(np.array(self.X[:, -2:]))
        # self.X[:, -2:] = normalized_x
        standardized_x, _, _, _, _ = self.standardize(np.array(self.X[:, -2:]))
        self.X[:, -2:] = standardized_x

        self.Y = np.array(self.Y)
        #self.Y, self.Y_min, self.Y_max, self.Y_mean = self.normalize(self.Y)
        self.Y, self.Y_min, self.Y_max, self.Y_mean, self.Y_std = self.standardize(self.Y)

    def normalize(self, x):
        #x_min = np.min(x) # across whole array
        x_min = np.min(x, axis=0) # across the columns -> 6 values
        x_max = np.max(x, axis=0)
        normalized_x_init = ((x - x_min) / (x_max - x_min)) # yields range [0, 1]

        # [0, 1] -> -0.5 -> [-0.5, 0.5] -> *2 -> [-1, 1]
        normalized_x = (normalized_x_init - 0.5) * 2 # TODO why we applied scaling?

        return normalized_x, x_min, x_max, np.mean(x, axis=0), np.std(x, axis=0)

    """
        ensure data is centered around the mean 
    """
    def standardize(self, x):
        x_min = np.min(x, axis=0)  # across the columns -> 6 values
        x_max = np.max(x, axis=0)

        nju= np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x_stardardized = (x - nju) / std

        return x_stardardized, x_min, x_max, nju, std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.Y[idx])

class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        # how many batches (not samples) in the dataset
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch >= len(self):
            raise StopIteration()

        # for this specific batch - where to start and where to end
        idx_start = self.idx_batch * self.idx_batch + self.idx_start
        idx_end = idx_start + self.batch_size

        x, y = self.dataset[idx_start:idx_end]
        self.idx_batch += 1

        return x, y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
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
        self.W: Variable = Variable(  # init with equal value isnt effective as the change will be the same
            # instead make it random
            value=np.random.uniform(low=-1, size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b: Variable = Variable(
            value=np.random.randn(out_features),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        W_dot_x = (self.W.value.T @ x.value[:, :, None])[:, :, 0] # or use .squeeze() instead of [:,:,0]
        self.output = Variable(W_dot_x + self.b.value)
        return self.output

    def backward(self):# have output, so here we calculate the amount of change -> gradient
        # y = f(x) + b
        # dy/db =
        # dy/dW =
        # dy/dx =

        self.b.grad += self.output.grad
        self.W.grad += self.x.value[:, :, None] @ self.output.grad[:, None, :]
        self.x.grad += (self.W.value @ self.output.grad[:, :, None])[:, :, 0]

class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    # sig(x) =  1./(1. + np.exp(-x.value))
    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            1./(1. + np.exp(-x.value))
        )
        return self.output

    # sig(x)/dx = sig(x)*(1-sig(x))
    def backward(self):
        self.x.grad += (1. - self.output.value) + self.output.value + self.output.grad # TODO why we add self.output here?

def sigmoid(x):
    return 1./(1. + np.exp(-x))

class LayerSwish():
    def __init__(self):
        self.x = None
        self.output = None

    # swish(x) =  x/(1. + np.exp(-x.value))
    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            x.value / (1. + np.exp(-x.value))
        )
        return self.output

    # swish(x)/dx = swish(x) + sig(x)(1-swish(x))
    def backward(self):
        self.x.grad += (self.output.value + sigmoid(self.x.value)*(1. - self.output.value)) + (self.output.value + self.output.grad)

class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            (x.value >= 0)*x.value
        )
        return self.output

    def backward(self):
        self.x.grad += (self.x.value >= 0 ).astype(np.float32) * self.output.grad

class LayerR2Score:
    def __init__(self, y_mean, std):
        self.y = None
        self.y_prim  = None
        self.score = None

        self.y_mean = y_mean
        self.std = std

    """
    R² = 1 → Perfect prediction
    R² = 0 → Model predicts no better than the mean    
    R² < 0 → Model is worse than just predicting the mean
     
    Copied from: ChatGPT - <https://chatgpt.com/>
    """
    # r2 = 1 - sum (y-y')^2 / sum (y-y_mean) ^2
    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        # reverse standardization and only then calculate r2
        y_curr_nonstd = y.value * self.std + self.y_mean
        y_prim_nonstd = y_prim.value * self.std + self.y_mean
        #y_mean =

        #r2 = 1 - np.sum((y.value-y_prim.value)**2) / np.sum((y.value-self.y_mean)**2)
        r2 = 1 - np.sum((y_curr_nonstd - y_prim_nonstd) ** 2) / np.sum((y_curr_nonstd - self.y_mean) ** 2)
        self.score = r2
        return self.score

class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    # mse = (1/n)* (y-y')^2
    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean((y.value-y_prim.value)**2)
        return loss

    # d mse/dx = -2 * (y-y')
    def backward(self):
        self.y_prim.grad += -2 * ( self.y.value - self.y_prim.value)

"""
Huber Loss is a loss function that's a mix between MSE and MAE (Mean Absolute Error). It's designed to be robust to outliers.
 δ (or γ in your case) is the hyperparameter that controls the "cutoff" between MSE-like behavior and MAE-like behavior.
 When error is small: behaves like MSE.
 When error is large: behaves like MAE — more tolerant of outliers.


"""
class LossHuber():
    def __init__(self, gamma):
        self.y = None
        self.y_prim  = None
        self.gamma = gamma

    # huber = when |y-y'| <= gamma -> 0.5 * (y-y')**
    #         when        > gamma  -> gamma * (|y-y'| - 0.5 * gamma)
    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        #diff = y.value-y_prim.value
        diff = y_prim.value - y.value

        diff_abs = np.abs(diff)

        ## cant directly check array, use np.lib instead
        #if diff_abs <= self.gamma:
        #    loss = 0.5 * diff**2
        #else:
        #    loss = self.gamma * (diff_abs - 0.5 * self.gamma)

        # first compute both options
        # then merge results based on filter condition
        err_sq = 0.5 * diff**2
        err_linear = self.gamma * (diff_abs - 0.5 * self.gamma)

        is_err_less = diff_abs <= self.gamma
        loss = np.mean(np.where(is_err_less, err_sq, err_linear))
#        loss = np.mean(np.where(is_err_less,  err_linear, err_sq))
        return loss

    # d huber/dx =  when |y-y'| <= gamma -> (y-y')
    #                           > gamma  -> gamma * (y-y')/ (|y-y'| + eps)
    def backward(self):
        #diff = self.y.value - self.y_prim.value # !!!NB order is important - this one reverses direction of loss, i.e. increases it
        diff = self.y_prim.value - self.y.value
        diff_abs = np.abs(diff)
        is_err_less = diff_abs <= self.gamma
        #if diff_abs <= self.gamma:
        #    self.y_prim.grad += diff
        #else:
        #    self.y_prim.grad += self.gamma * diff / ( diff_abs + 1e-8)

        grad_lq = self.y_prim.grad + diff
        grad_gt = self.y_prim.grad + self.gamma * diff / ( diff_abs + 1e-8) # (d mae/dx) * gamma

        self.y_prim.grad = np.where(is_err_less, grad_lq, grad_gt)

class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    # mae = (1/n) * |y' - y|
    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y_prim.value - y.value))
        return loss

    # d mae/dx = (y'-y)/|y'-y|
    def backward(self):
        self.y_prim.grad += (self.y_prim.value - self.y.value) / (np.abs(self.y_prim.value - self.y.value) + 1e-8)

class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=6, out_features=4),
            #LayerLinear(in_features=4*EMBEDD_DIM+2, out_features=4),
            #LayerSigmoid(),
            LayerSwish(),
            LayerLinear(in_features=4, out_features=8),
            #LayerSigmoid(),
            LayerSwish(),
            LayerLinear(in_features=8, out_features=2)
        ]
        self.embeddings: list[LayerEmbedding] = []  # embedding is a trainable (learnable) (distances between) vectors

        for x_cat_labels in dataset_full.labels:
            self.embeddings.append(LayerEmbedding(
                num_embeddings=len(x_cat_labels),
                embedding_dim=EMBEDD_DIM
            ))
        self.x_categorical_embs: list[Variable] = []
        self.x_concat: Variable = None

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
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
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

# For later
"""
one hot vector optimization
instead of storing sparse matrix with zeroes where feature belong to one class only, 
we store some useful features for specific features
[one hot vector]x[ embedding weight matrix]
"""
class LayerEmbedding: # for categorical values cant be directly calculate difference, so embedd vectors for identifying those
    def __init__(self, num_embeddings, embedding_dim):
        self.x_indexes = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_m = Variable(np.random.random((num_embeddings, embedding_dim)))
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x_indexes = x.value.squeeze().astype(np.int32) #any diff int or int32?
        self.output = Variable(np.array(self.emb_m.value[self.x_indexes, :])) # same as dot product with one-hot encoded X and Emb_w
        return self.output

    def backward(self):
        self.emb_m.grad[self.x_indexes, :] += self.output.grad


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = None
if USE_HUBER:
    loss_fn = LossHuber(HUBER_THRESHOLD)
else:
    loss_fn = LossMAE() if USE_MAE else LossMSE()

r2Score_fn = LayerR2Score(dataset_full.Y_mean, dataset_full.Y_std)

loss_plot_train = []
loss_plot_test = []
r2Scores = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, y in dataloader:

            #TODO catch where the empty batch is let through
            if x.shape[0] == 0 or y.shape[0] == 0:
                continue

            # Y' = Linear(sigma(Linear(sigma(Linear(x)))))
            y_prim = model.forward(Variable(value=x)) # prediction
            loss = loss_fn.forward(y=Variable(y), y_prim=y_prim) # how prediction compares to ground truth
            r2Score = r2Score_fn.forward(y=Variable(y), y_prim=y_prim) # how well model explains data

            losses.append(loss)
            r2Scores.append(r2Score)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()
                pass

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch}: R2: {r2Scores[-1]:.4f};  loss_train: {loss_plot_train[-1]:.4f} loss_test: {loss_plot_test[-1]:.4f}')

    if epoch % 10 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        #plt.show()

        plt.show(block=False)
        #plt.close(fig) #accumlates heap overuse