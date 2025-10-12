import matplotlib.pyplot as plt
import numpy as np

X = np.array([#todo add more for better prediction
    [2021, 112_000], # CAR1
    [2012, 141_000], # CAR2
    [2006, 243_000], # CAR3
    [2012, 225_000], # CAR3
    [2020, 185_000]  # CAR3
]) # + 2000 (5x2)

Y = np.array([#prices
    [42_500], # dot product is 2d operation
    [18_900],
    [7350],
    [13_899],
    [29_490]
]) # * 1000 (5x1)

#change this to some transformation matrix
W_1 = np.array([
    [0,0,0,0],
    [0,0,0,0]
])#(2,4)
W_1 = np.random.randn(2,4)

# W dont have add dim as they are exact the same for every car
# y' = W_1 @ X[0] +b_1 # (2x4)@(5x2 ,1 (additional dimension)) + (4,) -> (5,4) # 5 cars projection to 4 dimensions
b_1 = np.array([0.,0.,0.,0.]) # (4, )

#a= np.array([1,2,3]) # (3,)
#b= [[1],[2],[3]] # 3 ROWS 1 COL - added extra dimension
#b = a[:, None] # add additional dimension
#b = a.expand_dims(axis=1) # alternative to prev (3,1)
#a = b [: , 0] # remove dimension (3,)  => [1,2,3]

# y'' = W_2 @ y'  scalar result
W_2 = np.random.randn(4,1)
b_2 = np.zeros(1,) # 1d

# Linear(x,W,b) = W . x + b
def linear(W, b, x): # connect cars with features
    out = W.T @ x[:, :, None] # (B, in_features, 1extra dim); in derivative no need for .T
    out = out[:,:,0]+b # W = (in_features, out_features)
    return out
    #return 0 #TODO

# sig(x) = 1 / ...
def sigmoid(x):
    out = (1. / (1.+np.exp(-x)))
    return out

# y' = Model (x, W_1, b_1, W_2, b_2) = Linear(sig(Linear(x, W_1, b_1)), W_2, b_2)
def model(x, W_1, b_1, W_2, b_2):
    out = linear(W_1, b_1, x)
    out = sigmoid(out)
    out = linear(W_2, b_2, out)
    return out

def loss_mae(y_prim, y):
    out = np.mean(np.abs(y_prim -y))
    return out
    #return 0 #TODO

def dW_linear(W, b, x): #  (W@x+b)/dW = x
    return x #TODO

def db_linear(W, b, x): #(W@x+b) / dB
    return 1 #TODO

def dx_linear(W, b, x): #(W@x+b) / dW
    return W #TODO

X_mean = np.mean(X,axis=0) # keep
X_std= np.std(X, axis=0) # keep for future calculation if new data will be added
X = (X-X_mean)/X_std

Y_mean = np.mean(Y,axis=0) # keep
Y_std= np.std(Y, axis=0) # keep to be able to reverse cnormalization for analyzing
Y = (Y-Y_mean)/Y_std

"""
feature normalization is required to use with exp
 e.g. minmax, etc
 the better approach is  to use normal std
"""
def dx_sigmoid(x):
    out = np.exp(-x)/(1. + np.exp(-x))**2
    return out #TODO

def dy_prim_loss_mae(y_prim, y):
    out = (y_prim - y)/ (np.abs(y_prim-y) + 1e-8)
    return out #TODO

def dW_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    out_1 = dy_prim_loss_mae(y_prim, y) # 5x1
    out_2 = dx_linear( W_2, b_2, x=sigmoid(linear(W_1, b_1, x))) # 4x1
    out_3 = dx_sigmoid(x=linear(W_1, b_1, x)) # 5x4
    out_4 = dW_linear(W_1, b_1, x) # 5x2
    out_5 = (out_2 @ out_1[:, :, None])[:, :, 0]  # repeat the same weights for each cars and return original dimensions
    out_6 = out_5 * out_3
    out_7 = out_4[:,:, None] @ out_6[:, None,:]
    dW_1 = np.mean(out_7, axis=0)
    return dW_1 #TODO


def db_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    out_1 = dy_prim_loss_mae(y_prim, y) # 5x1
    out_2 = dx_linear( W_2, b_2, x=sigmoid(linear(W_1, b_1, x))) # 4x1
    out_3 = dx_sigmoid(x=linear(W_1, b_1, x)) # 5x4
    out_4 = db_linear(W_1, b_1, x) # 5x2
    out_5 = (out_2 @ out_1[:,:, None])[:,:,0] # repeat the same weights for each cars and return original dimensions
    out_6 = out_5 * out_3
    db_1 = np.mean(out_6 * out_4, axis=0)# without axis calculates 1 scalar value; with axis calculates mean for each column
    assert b_1.shape == db_1.shape # safety check
    return db_1 #TODO

def dW_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    out_1 = dy_prim_loss_mae(y_prim, y) #5x1
    _sigmoid=sigmoid(linear(W_1, b_1, x))  # 5x4
    out_2 = dW_linear(W_2, b_2, x=_sigmoid)
    out_3 = out_1[:, :,None] @ out_2[:, None, :]
    dW_2 = np.mean(out_3, axis=0).T # transpose works on 2 dimensions
    assert W_2.shape == dW_2.shape
    return dW_2  # TODO

def db_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    out_1 = dy_prim_loss_mae(y_prim, y)
    _sigmoid=sigmoid(linear(W_1, b_1, x))  # 4x1
    out_2 = db_linear(W_2, b_2, x=_sigmoid)
    db_2 = np.mean(out_1 * out_2, axis=0)
    assert b_2.shape == db_2.shape
    return db_2 #TODO

# MAE:derivatives always points to minimum, so even if loss jumps overs 0 to negative side, the next step will became abs(smaller)
# is good for dirty dataset
# MSE
# is good for clean dataset - will not step back, like in  case of MAE
learning_rate = 1e-2
losses = []
for epoch in range(1000):

    Y_prim = model(X, W_1, b_1, W_2, b_2)
    loss = loss_mae(Y_prim, Y)

    #TODO SGD
    #dW_1 = np.sum(dW_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, y))
    db_1 = db_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y)
    db_2 = db_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y)
    dW_1 = np.sum(dW_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    dW_2 = np.sum(dW_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))

    b_2 -= db_2 * learning_rate
    b_1 -= db_1 * learning_rate

    Y_prim_real = Y_prim * Y_std + Y_mean
    Y_real = Y* Y_std + Y_mean

    print(f'Y_prim: {Y_prim}; Y: {Y};')
    print(f'Y_prim_real:{Y_prim_real} Y_real:{Y_real}')
    print(f'loss: {loss}')
    losses.append(loss)

plt.plot(losses)
plt.show()