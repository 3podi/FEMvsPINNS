from pyDOE import lhs
import jax.numpy as jnp
import matplotlib.pyplot as plt

def sample_points(low_b=0.,up_b=1.,num_domain=256):
    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(2, num_domain)
    boundary_points = lb + (ub-lb)*lhs(1, 2)

    return domain_points, boundary_points


def points_to_hashable(points, precision=5):
    """
    Convert points into a hashable format (tuples).
    """
    return {tuple(jnp.round(point, precision)) for point in points}

def filter_points(sampled_points, validation_points):
    """
    Filter sampled points to exclude those in the validation set.
    
    Arguments:
    - sampled_points: Array of points to filter.
    - validation_points: Validation set of points.
    
    Returns:
    - Filtered array of sampled points.
    """
    validation_set = points_to_hashable(validation_points)
    filtered_points = [point for point in sampled_points if tuple(jnp.round(point, 5)) not in validation_set]
    return jnp.array(filtered_points)

def sample_training_points(validation_points, low_b=0.,up_b=1.,num_domain=256):
    """
    Sample training points excluding validation points.
    """

    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(2, num_domain)
    boundary_points = lb[0] + (ub[0]-lb[0])*lhs(1, 2)

    val_domain_points =validation_points[0]

    domain_points = filter_points(domain_points, val_domain_points)
    return domain_points, boundary_points


def plot_losses(train_data, val_data):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten() 

    # Plot training and validation metrics
    for idx, key in enumerate(train_data.keys()):
        axes[idx].plot(train_data[key], marker='o', linestyle='-', label='Train')
        axes[idx].plot(val_data[key], marker='s', linestyle='--', label='Validation')
        axes[idx].set_title(key)
        axes[idx].set_xlabel('Epochs')
        axes[idx].set_ylabel('Value')
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()