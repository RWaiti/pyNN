# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Imports

# %%
import numpy as np

# %% [markdown]
# ## initialize_parameters

# %%
def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
        
    for l in range(1, L+1):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters

# %% [markdown]
# ## activation_funtion

# %%
def activation_funtion(Z, activation):
    if activation == "relu":
        A = np.maximum(np.zeros(1), Z)
        print("activation:", activation)
        return A
        
    if activation == "sigmoid":
        A = 1 / (1-np.exp(-Z) + 1e-32)
        print("activation:", activation)
        return A
    
    print("activation: linear")
    return Z

# %% [markdown]
# ## forward_propagation

# %%
def forward_propagation(X, parameters):
    cache = {}
    
    L = int(len(parameters)/2)
    l = 1
    cache['Z'+str(1)] = np.dot(parameters['W'+str(1)], X) + parameters['b'+str(1)]
    cache['A'+str(1)] = activation_funtion(cache['Z'+str(1)], "relu")
    
    for l in range(2,L):
        cache['Z'+str(l)] = np.dot(parameters['W'+str(l)], cache['A'+str(l-1)]) + parameters['b'+str(l)]
        cache['A'+str(l)] = activation_funtion(cache['Z'+str(l)], "relu")
    l += 1
    cache['Z'+str(l)] = np.dot(parameters['W'+str(l)], cache['A'+str(l-1)]) + parameters['b'+str(l)]
    cache['A'+str(l)] = activation_funtion(cache['Z'+str(l)], "sigmoid")

    return cache

# %% [markdown]
# ## back_propagation

# %%


# %% [markdown]
# ## Test

# %%
layers_dims = [3,6,2]
X = np.random.randn(layers_dims[0],1)
parameters = initialize_parameters(layers_dims)
print(forward_propagation(X,parameters))


# %%
layers_dims = [3,7,6,2]
X = np.random.randn(layers_dims[0],1)
parameters = initialize_parameters(layers_dims)
print(forward_propagation(X,parameters))


