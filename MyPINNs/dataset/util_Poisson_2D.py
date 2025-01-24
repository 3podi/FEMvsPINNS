from pyDOE import lhs
import jax.numpy as jnp
import matplotlib.pyplot as plt

def sample_points(low_b=0.,up_b=1.,num_domain=2000):
    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(2, num_domain)
    boundary_points = lb + (ub-lb)*lhs(2, 250)

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

def sample_training_points(validation_points, low_b=0.,up_b=1.,num_domain=2000):
    """
    Sample training points excluding validation points.
    """

    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(2, num_domain)
    boundary_points = lb + (ub-lb)*lhs(2, 250)

    val_domain_points =validation_points[0]
    val_boundary_points = validation_points[1]

    domain_points = filter_points(domain_points, val_domain_points)
    boundary_points = filter_points(boundary_points,val_boundary_points)
    return domain_points, boundary_points

