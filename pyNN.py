# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Only using tensorflow to import a dataset

# %%
import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


# %%
side = 5
start = np.random.random_integers(low=0, high=x_train.shape[0], size=(1,))[0]
fig, ax = plt.subplots(side, side)
for a in range(side):
    for b in range(side):
        ax[a, b].axes.xaxis.set_visible(False)
        ax[a, b].axes.yaxis.set_visible(False)
        ax[a, b].imshow(x_train[2 * a + b + start, :, :], cmap='gray')
plt.show()

# %% [markdown]
# ## Code
# %% [markdown]
# ## Import

# %%
import numpy as np

# %% [markdown]
# ## flatten and reshape

# %%
print("x_train:", x_train.shape)
x_train = x_train.reshape(x_train.shape[0], -1).T # flatten
print("x_train flatten:", x_train.shape)

print("x_test:", x_test.shape)
x_test = x_test.reshape(x_test.shape[0], -1).T # flatten
print("x_test flatten:", x_test.shape)

print("y_train:", y_train.shape)
y_train = y_train.reshape(y_train.shape[0],1).T # reshape
print("y_train:", y_train.shape)

print("y_test:", y_test.shape)
y_test = y_test.reshape(y_test.shape[0],1).T # reshape
print("y_test:", y_test.shape)

# %% [markdown]
# ## One Hot

# %%
y_train_onehot = np.zeros((len(np.unique(y_train)), y_train.shape[1]), dtype=np.float32)
y_test_onehot = np.zeros((len(np.unique(y_test)), y_test.shape[1]), dtype=np.float32)

print(y_train_onehot)
print(y_train_onehot.shape)
print(y_train.ravel())

y_train_onehot[y_train, y_train] = 1
y_test_onehot[y_test, y_test] = 1

print(y_train_onehot) 

# %% [markdown]
# ## Normalization

# %%
x_train = x_train/255
x_test = x_test/255

# %% [markdown]
# ## initialize_parameters

# %%
def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
        
    for l in range(1, L+1):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1], dtype=np.float32) * np.sqrt(2./layers_dims[l-1])
        parameters['b'+str(l)] = np.ones((layers_dims[l], 1), dtype=np.float32)
    
    return parameters

# %% [markdown]
# ## activation_funtion

# %%
def activation_funtion(Z, activation):
    if activation == "relu":
        return np.maximum(np.zeros(1), Z)
        
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-Z))
    
    return Z

# %% [markdown]
# ## forward_propagation

# %%
def forward_propagation(X, parameters, activation):
    cache = {}
    L = len(parameters) // 2
    cache['A0'] = X
    
    for l in range(1,L):
        cache['Z'+str(l)] = np.dot(parameters['W'+str(l)], cache['A'+str(l-1)]) + parameters['b'+str(l)]
        cache['A'+str(l)] = activation_funtion(cache['Z'+str(l)], activation[l-1])

    l += 1
    
    cache['Z'+str(l)] = np.dot(parameters['W'+str(l)], cache['A'+str(l-1)]) + parameters['b'+str(l)]
    cache['A'+str(l)] = activation_funtion(cache['Z'+str(l)], activation[l-1])
    
    return cache

# %% [markdown]
# ## compute_cost

# %%
def compute_cost(AL,Y, loss = "cross_entropy"):
    m = Y.shape[1]
    
    AL = AL.T
    if loss == "cross_entropy":
        epsilon = 1e-7
        cost = -(1/m) * np.sum(np.multiply(np.log(AL + epsilon), Y) + np.multiply(np.log(1-AL + epsilon), 1-Y))
        return np.squeeze(cost)


# %%
def activation_derivative(Z, activation):
    if activation == "relu":
        dZ = np.ones(Z.shape)
        return (dZ >= 0).astype(int)
        
    if activation == "sigmoid":
        return np.multiply(activation_funtion(Z, "sigmoid"), (1- activation_funtion(Z, "sigmoid")))
    
    return Z

# %% [markdown]
# ## back_propagation

# %%
def back_propagation(X, Y, parameters, cache, activation):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[1]
    
    grads["dZ" + str(L)] = cache['A' + str(L)] - Y
    grads["dW" + str(L)] = (1/m) * np.dot(grads["dZ" + str(L)], cache['A' + str(L-1)].T)
    grads["db" + str(L)] = (1/m) * np.sum(grads["dZ" + str(L)], axis= 1, keepdims= True)
    
    for l in range(L-1,1,-1):
        grads["dZ" + str(l)] = np.multiply(np.dot(parameters["W" + str(l+1)].T, grads["dZ" + str(l+1)]), activation_derivative(cache["Z" + str(l)],activation[l-1]))        
        grads["dW" + str(l)] = (1/m) * np.dot(grads["dZ" + str(l)], cache['A' + str(l-1)].T)
        grads["db" + str(l)] = (1/m) * np.sum(grads["dZ" + str(l)], axis= 1, keepdims= True)
    l -= 1
    
    grads["dZ" + str(l)] = np.multiply(np.dot(parameters["W" + str(l+1)].T, grads["dZ" + str(l+1)]), activation_derivative(cache["Z" + str(l)],activation[l-1]))
    grads["dW" + str(l)] = (1/m) * np.dot(grads["dZ" + str(l)], cache['A' + str(l-1)].T)
    grads["db" + str(l)] = (1/m) * np.sum(grads["dZ" + str(l)], axis= 1, keepdims= True)
    
    return grads

# %% [markdown]
# ## update_parameters

# %%
def update_parameters(grads, paramaters, lr=0.01):
    
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - np.multiply(grads['dW'+str(l)], lr)
        parameters['b'+str(l)] = parameters['b'+str(l)] - np.multiply(grads['db'+str(l)], lr)
    
    return parameters

# %% [markdown]
# ## prediction

# %%
def prediction(X, Y, parameters, activation):
    cache = forward_propagation(X, parameters, activation)
    
    cost = compute_cost(cache['A' + str(len(parameters) // 2)].T, Y)
    
    pred = cache['A' + str(len(parameters) // 2)] - Y
    
    print(pred)
    
    return "train cost: " + str(cost)


# %%
layers_dims = [x_train.shape[0],64,64,10]
activation = ["relu","relu","sigmoid"]
# loss_funtion = "cross_entropy"

print(x_train.shape)
print(y_train.shape)

parameters = initialize_parameters(layers_dims)

iter = 100

for i in range(iter):
    cache = forward_propagation(x_train, parameters, activation)

    
    if i % 10 == 0:
        cost = compute_cost(cache['A' + str(len(parameters) // 2)].T, y_train_onehot)
        print("cost train: " + str(cost), end= " ")
        cache_test = forward_propagation(x_test, parameters, activation)
        cost = compute_cost(cache_test['A' + str(len(parameters) // 2)].T, y_test_onehot)
        print("cost test: " + str(cost))
        
    grads = back_propagation(x_train, y_train_onehot, parameters, cache, activation)

    parameters = update_parameters(grads, parameters, 0.001)

print(parameters)


# %%
for i in y_train:
    print(i)

for i in y_train_onehot:
    print(i)
    
for i in cache['A' + str(len(parameters) // 2)]:
    print(i)


# %%
layers_dims = [32,18,7,2] # 3NN
activation = ["relu","relu","sigmoid"] # 3 Activation to 3NN
# loss_funtion = "cross_entropy"

X = np.random.randn(layers_dims[0],15) * 10
Y = np.array([[0, 1, 0.2, 0.5, 0.2, 0, 1, 0.2, 0.5, 0.2, 0, 1, 0.2, 0.5, 0.2]])

parameters = initialize_parameters(layers_dims)

iter = 1500

for i in range(iter):
    cache = forward_propagation(X, parameters, activation)

    cost = compute_cost(cache['A' + str(len(parameters) // 2)].T, Y)
    if i % 10 == 0:
        print(cost)
    grads = back_propagation(X, Y, parameters, cache, activation)

    parameters = update_parameters(grads, parameters, 0.02)

print(parameters)


