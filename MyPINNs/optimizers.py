import jax
import jax.numpy as jnp

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer with the given hyperparameters.
        
        Arguments:
            learning_rate: The learning rate for the optimizer (default 0.001)
            beta1: Exponential decay rate for the first moment estimate (default 0.9)
            beta2: Exponential decay rate for the second moment estimate (default 0.999)
            epsilon: A small constant for numerical stability (default 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init(self, params):
        """
        Initialize the Adam optimizer state (momentum and velocity).
        
        Arguments:
            params: List of model parameters (weights and biases)
        
        Returns:
            A dictionary containing the optimizer state.
        """
        
        return {
            'momentum': jax.tree_map(lambda p: jnp.zeros_like(p), params),
            'velocity': jax.tree_map(lambda p: jnp.zeros_like(p), params),
            'beta1_power': jnp.array(self.beta1),
            'beta2_power': jnp.array(self.beta2),
        }

    def update(self, params, grads, state):
        """
        Perform a single optimization step using the Adam optimizer.
        
        Arguments:
            params: List of model parameters (weights and biases)
            grads: List of gradients corresponding to the parameters
            state: The current optimizer state (momentum and velocity)
        
        Returns:
            updated_params: List of updated model parameters
            updated_state: Updated optimizer state
        """
        def apply_update(param, grad, m, v):
            # Update biased first and second moment estimates
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - state['beta1_power'])
            v_hat = v / (1 - state['beta2_power'])
            
            # Parameter update
            param = param - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
            return param, m, v

        updated_params, (updated_momentum, updated_velocity) = jax.tree_map(
            lambda p, g, m, v: apply_update(p, g, m, v),
            params, grads, state['momentum'], state['velocity']
        )

        # Update the optimizer state
        updated_state = {
            'momentum': updated_momentum,
            'velocity': updated_velocity,
            'beta1_power': state['beta1_power'] * self.beta1,
            'beta2_power': state['beta2_power'] * self.beta2,
        }

        return updated_params, updated_state
    

class Adam2:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init(self, params):
        m = jax.tree_map(jnp.zeros_like, params)
        v = jax.tree_map(jnp.zeros_like, params)
        
        return {'momentum': m, 
                'velocity': v,
                'beta1_power': jnp.array(self.beta1),
                'beta2_power': jnp.array(self.beta2),
                }

    def update(self, params, grads, state):

        m = state['momentum']
        v = state['velocity']

        m = jax.tree_map(lambda g, m: self.beta1 * m + (1 - self.beta1) * g, grads, m)
        v = jax.tree_map(lambda g, v: self.beta2 * v + (1 - self.beta2) * jnp.square(g), grads, v)

        m_hat = jax.tree_map(lambda m: m / (1 - state['beta1_power']), m)
        v_hat = jax.tree_map(lambda v: v / (1 - state['beta2_power']), v)

        params = jax.tree_map(lambda p, m_hat, v_hat: p - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon), params, m_hat, v_hat)
        state = {'momentum': m, 'velocity': v, 'beta1_power': state['beta1_power'] * self.beta1, 'beta2_power': state['beta2_power'] * self.beta2}
        
        return params, state
            

class AdamW:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        # Initialize hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
    def init_state(self, params):
        # Initialize optimizer state for each parameter
        state = {}
        state['momentum'] = [jnp.zeros_like(p) for p in params]
        state['velocity'] = [jnp.zeros_like(p) for p in params]
        state['beta1_power'] = jnp.ones(())
        state['beta2_power'] = jnp.ones(())
        return state

    def apply_update(self, params, grads, state):
        updated_params = []
        updated_state = {}
        
        # Extract momentum and velocity from the state
        momentum = state['momentum']
        velocity = state['velocity']
        
        # Apply AdamW update for each parameter
        for p, g, m, v in zip(params, grads, momentum, velocity):
            # Update the first and second moment estimates
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            
            # Apply bias correction
            m_hat = m / (1 - state['beta1_power'])
            v_hat = v / (1 - state['beta2_power'])
            
            # Compute parameter update with weight decay (decoupled L2 regularization)
            param_update = self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
            
            # Apply weight decay (decoupled)
            param_update += self.learning_rate * self.weight_decay * p

            # Update the parameters
            updated_params.append(p - param_update)
            
            # Update the momentum and velocity
            updated_state['momentum'] = [m for m in state['momentum']]
            updated_state['velocity'] = [v for v in state['velocity']]
        
        # Update the beta powers for the bias correction
        updated_state['beta1_power'] = state['beta1_power'] * self.beta1
        updated_state['beta2_power'] = state['beta2_power'] * self.beta2

        return updated_params, updated_state