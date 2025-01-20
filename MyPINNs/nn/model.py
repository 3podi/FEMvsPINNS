import jax.numpy as jnp
import jax
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
  

class Embedder:
    def __init__(self, input_dims, include_input=True, max_freq_log2=4, num_freqs=6, log_sampling=True, periodic_fns=[jnp.sin, jnp.cos]):
        """
        Initializes the embedding layer.

        Args:
            input_dims (int): Number of input dimensions.
            include_input (bool): Whether to include the original input in the output.
            max_freq_log2 (int): Maximum frequency for log-scale sampling.
            num_freqs (int): Number of frequency bands.
            log_sampling (bool): Whether to use logarithmic frequency sampling.
            periodic_fns (list): List of periodic functions (e.g., [sin, cos]).
        """
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Generates the embedding functions based on the configuration."""
        embed_fns = []
        out_dim = 0

        # Optionally include the input itself
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        # Create frequency bands
        if self.num_freqs > 0:
            if self.log_sampling:
                freq_bands = jnp.logspace(0.0, self.max_freq_log2, num=self.num_freqs, base=2.0)
            else:
                freq_bands = jnp.linspace(2.0**0.0, 2.0**self.max_freq_log2, num=self.num_freqs)

            # Add sin and cos embeddings for each frequency
            for freq in freq_bands:
                for p_fn in self.periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                    out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """
        Applies the embedding functions to the input.

        Args:
            inputs (jax.numpy.ndarray): Input tensor of shape [B, input_dims]

        Returns:
            jax.numpy.ndarray: Embedded tensor of shape [B, out_dim]
        """
        return jnp.concatenate([fn(inputs) for fn in self.embed_fns], axis=-1)
    
  
def ANN_emb(params, x, embedder):
    """
    MLP function with an embedder.
    
    Args:
        params (list): List of weight matrices and biases.
        x (jax.numpy.ndarray): Input tensor of shape [B, D].
        embedder (Embedder): Embedder object to preprocess inputs.
    
    Returns:
        jax.numpy.ndarray: Model output.
    """
    x = jnp.atleast_2d(x)
    # Embed the second column of x (x[:, 1])
    x_embedded = embedder.embed(x[:, 1:2])  # Shape: [N, features]
    x_embedded = jax.lax.stop_gradient(x_embedded)
    # Concatenate with the first column of x (x[:, 0]) to get shape [N, 1 + features]
    x_embedded = jnp.concatenate([x[:, 0:1], x_embedded], axis=-1)

    # Forward pass through MLP
    layer = x_embedded.T  # Transpose to match weight shape
    num_layers = len(params) // 2

    for i in range(num_layers):
        W, b = params[2 * i], params[2 * i + 1]
        layer = jnp.dot(W, layer) + b
        if i < num_layers - 1:
            layer = jnp.maximum(0, layer)  # ReLU activation

    return layer.T  # Transpose back to [B, output_dim]


if __name__ == '__main__':
   
   layout = [3, 12, 12, 1]
   params = initialize_params(layout)

   for w in params:
      print('Weights shape: ', params.shape)