from pyDOE import lhs
import jax.numpy as jnp

def sample_points(low_b,up_b,num_domain,num_bound,num_ini):
    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(2, num_domain)
    boundary = lb[0] + (ub[0]-lb[0])*lhs(1, num_bound)
    init = lb[1] + (ub[1]-lb[1])*lhs(1, num_ini)

    return domain_points, boundary, init


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

def sample_training_points(low_b,up_b,num_domain,num_bound,num_ini,validation_points, init_only=False):
    """
    Sample training points excluding validation points.
    """

    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(2, num_domain)
    boundary = lb[0] + (ub[0]-lb[0])*lhs(1, num_bound)
    init = lb[1] + (ub[1]-lb[1])*lhs(1, num_ini)

    if init_only:
        val_init = validation_points[0]
        init = filter_points(init, val_init)
        return init
    else:
        val_domain_points =validation_points[0]
        val_boundary = validation_points[1]
        val_init = validation_points[2]

        domain_points = filter_points(domain_points, val_domain_points)
        boundary = filter_points(boundary, val_boundary)
        init = filter_points(init, val_init)
        return domain_points, boundary, init

