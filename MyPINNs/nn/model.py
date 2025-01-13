import jax.numpy as jnp
import numpy as np

def initialize_params(architecture_layout):
    """
    Initialization of weights given layout of a MLP network
    with Glorot (Xavier) initialization for weights.
    
    Input:
        architecture_layout: list with number of nodes in each layer [list]
    """
    
    np.random.seed(0)  # For reproducibility
    params = list()
    
    for i in range(len(architecture_layout) - 1):
        fan_in = architecture_layout[i]  # Number of input units
        fan_out = architecture_layout[i+1]  # Number of output units
        
        # Glorot Initialization for weights
        W = np.random.randn(fan_out, fan_in) * np.sqrt(2. / (fan_in + fan_out))
        
        # Bias initialization (zeros)
        b = np.zeros((fan_out, 1))
        
        params.append(W)
        params.append(b)
    
    return params

def relu(x):
    """ReLU activation function"""
    return jnp.maximum(0, x)

def ANN(params, x):
  """
  MLP function
  Input:
    params: list of weight matrices and biases previously initialized [list]
    x: input shape [B,D]
  """

  layer = x.T
  num_layers = int(len(params) / 2 + 1)
  weights = params[0::2]
  biases = params[1::2]
  for i in range(num_layers - 1):
    layer = jnp.dot(weights[i], layer) - biases[i]
    if i < num_layers - 2:
      layer = relu(layer)
  return layer.T


if __name__ == '__main__':
   
   layout = [3, 12, 12, 1]
   params = initialize_params(layout)

   for w in params:
      print('Weights shape: ', params.shape)