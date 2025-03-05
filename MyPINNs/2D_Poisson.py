import os
import jax, time, pickle
import jax.numpy as jnp
import numpy as onp
from functools import partial
import json

import numpy.random as npr

from nn.model import initialize_params
from nn.model import ANN2 as ANN
from optimizers import Adam2 as Adam
from lr_schedulers import LinearWarmupCosineDecay
from dataset.util_Poisson_2D import sample_points, sample_training_points


#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# Analytic solution of the 2D Poisson equation
@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
@jax.jit
def analytic_sol(xs,ys):
    out = (xs**2) * ((xs-1)**2) * ys * ((ys-1)**2)
    return out


@jax.jit
def analytic_sol1(xs,ys):
    out = (xs**2) * ((xs-1)**2) * ys * ((ys-1)**2)
    return out

# Derivatives for the Neumann Boundary Conditions
@partial(jax.vmap, in_axes=(None, 0, 0,), out_axes=(0, 0, 0))
@jax.jit
def neumann_derivatives(params,xs,ys):
    u = lambda x, y: ANN(params, jnp.stack((x, y)),dim=2)
    du_dx_0 = jax.jvp(u,(0.,ys),(1.,0.))[1]
    du_dx_1 = jax.jvp(u,(1.,ys),(1.,0.))[1]
    du_dy_1 = jax.jvp(u,(xs,1.),(0.,1.))[1]
    return du_dx_0, du_dx_1, du_dy_1

# PDE residual for 2D Poisson
@partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x, y):
    H1 = jax.hessian(u, argnums=0)(x,y)
    H2 = jax.hessian(u, argnums=1)(x,y)
    lhs = H1+H2
    rhs = 2*((x**4)*(3*y-2) + (x**3)*(4-6*y) + (x**2)*(6*(y**3)-12*(y**2)+9*y-2) - 6*x*((y-1)**2)*y + ((y-1)**2)*y )
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x, y: ANN(params, jnp.stack((x, y)),dim=2), points[:, 0], points[:, 1])**2)


@partial(jax.jit, static_argnums=0)
def pde_true(analytic_sol,params, points):
    return jnp.mean((ANN(params, jnp.stack((points[:, 0], points[:, 1]), axis=1),dim=2) - analytic_sol(points[:, 0], points[:, 1]) )**2)

@jax.jit
def boundary_dirichlet(params, points): # u(x,0) = 0
    return jnp.mean((ANN(params, jnp.stack((points[:,0],jnp.zeros_like(points[:,1])), axis=1),dim=2))**2) 

@partial(jax.jit, static_argnums=0) # du/dx(0,y) = 0, du/dx(1,y) = 0, du/dy(x,1) = 0
def boundary_neumann(neumann_derivatives, params, points):
    du_dx_0, du_dx_1, du_dy_1 = neumann_derivatives(params,points[:,0],points[:,1])
    return jnp.mean((du_dx_0)**2) + jnp.mean((du_dx_1)**2) + jnp.mean((du_dy_1)**2)
#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,4))
def training_step(params, opt, opt_state, val_points, neumann_derivatives):
    domain_points, boundary_points = sample_training_points(val_points, low_b=[0.,0.], up_b=[1.,1.])

    domain_points = jax.device_put(domain_points)
    boundary_points = jax.device_put(boundary_points)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    boundary_dirichlet(params, boundary_points) +
                                                    boundary_neumann(neumann_derivatives, params, boundary_points))(params)
    
    params, opt_state = opt.update(params, grad, opt_state)
    return params, opt_state, loss_val

@jax.jit
def validation_step(params, val_points):
    val_domain_xs, val_boundary_xs = val_points

    # Define a loss function for delayed computation
    loss_fn = lambda p: (
        pde_residual(p, val_domain_xs) + 
        boundary_dirichlet(p, val_boundary_xs) +
        boundary_neumann(neumann_derivatives, p, val_boundary_xs)
    )

    return loss_fn(params)


def train_loop(params, adam, opt_state, num_epochs, val_points, n_patience, validate_every=10, lr_scheduler=None, neumann_derivatives=neumann_derivatives):
    train_losses = []
    val_losses = []
    best_loss = 3000
    patience = n_patience
        
    for epoch in range(num_epochs):
        # Lr scheduler step
        lr = lr_scheduler.get_lr()
        adam.learning_rate = lr

        # Perform a training step
        params, opt_state, loss_train = training_step(params, adam, opt_state, val_points, neumann_derivatives)
        train_losses.append(loss_train.item())
        
        # Validation step (every `validate_every` epochs)
        if (epoch + 1) % validate_every == 0:
            loss_val = validation_step(params, val_points)  # Compute validation loss
            val_losses.append(loss_val.item())
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
            if best_loss - val_losses[-1] > 0.00001:
                best_loss = val_losses[-1]  # Update best loss
                patience = n_patience  # Reset patience
            else:
                patience -= 1
            
            if patience == 0:
                print('Early stopping the training, best val_loss: ', best_loss)
                break
            
    
    return train_losses, val_losses, params, opt_state

def main():
    print('Device: ', jax.default_backend())

    lr = 1e-3
    total_epochs = 20000
    validation_freq = 50

    val_domain_points, val_boundary = sample_points(low_b=[0.,0.], up_b=[1.,1.])
    
    val_domain_points = jax.device_put(val_domain_points)
    val_boundary = jax.device_put(val_boundary)

    #----------------------------------------------------
    # Define architectures list
    #----------------------------------------------------
    architecture_list = [[2,20,1],[2,60,1],[2,20,20,1],[2,60,60,1],[2,20,20,20,1],[2,60,60,60,1]]#,[20,20,20,20,1],[60,60,60,60,1]]
    #----------------------------------------------------
    # Train PINN
    #----------------------------------------------------
    # Train model 10 times and average over the times
    u_results = dict({})
    times_adam, times_eval, l2_rel, var, arch  = None, None, None, None, None
    print('Start training')
    for feature in architecture_list:
        print('Architecture: ', feature)
        times_adam_temp = []
        times_eval_temp = []
        l2_errors = []
        for _ in range(1):
            #----------------------------------------------------
            # Initialize Model
            #----------------------------------------------------
            params = initialize_params(feature)
            params = jax.device_put(params)

            #----------------------------------------------------
            # Initialise Optimiser
            #----------------------------------------------------
            optimizer = Adam(lr)
            opt_state = optimizer.init(params)
            lr_scheduler = LinearWarmupCosineDecay(warmup_epochs=100,total_epochs=total_epochs,base_lr=lr, min_lr=lr*1e-1)

            #----------------------------------------------------
            # Start Training
            #----------------------------------------------------
            start_time = time.time() 
            train_losses, val_losses, params, opt_state, = train_loop(params, optimizer, opt_state, total_epochs, n_patience=5, validate_every=validation_freq, lr_scheduler=lr_scheduler, val_points=[val_domain_points,val_boundary])
            adam_time = time.time()-start_time
            times_adam_temp.append(adam_time)
            #print("Adam training time: ", adam_time)

            
            with open("./Eval_Points/2D_Poisson_eval-points.json", 'r') as f:
                domain_points = json.load(f)
                domain_points = jnp.array(domain_points)
                domain_points = jax.device_put(domain_points)

            start_time3 = time.time()
            u_approx = ANN(params, jnp.stack((domain_points[:,0], domain_points[:,1]),axis=1),dim=2).squeeze()
            times_eval_temp.append(time.time()-start_time3)

            u_true = analytic_sol(domain_points[:,0],domain_points[:,1]).squeeze()
            run_accuracy = (onp.linalg.norm(u_approx - u_true))/onp.linalg.norm(u_true)
            l2_errors.append(run_accuracy)

        y_gt = u_true.tolist()
        domain_pts = domain_points.tolist()
        y_results = u_approx.tolist()
        times_adam, times_eval, l2_rel, var, arch = onp.mean(times_adam_temp), onp.mean(times_eval_temp), onp.mean(jnp.array(l2_errors)).tolist(), onp.var(jnp.array(l2_errors)).tolist(), feature

        results = dict({'domain_pts': domain_pts,
                        'y_results': y_results,
                        'y_gt': y_gt})

        evaluation = dict({'arch': arch,
            'times_adam': times_adam,
            'times_eval': times_eval,
            'l2_rel': l2_rel,
            'var': var})

        save_dir = './MyPINNS_results/2D-Poisson-PINN'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        architecture_name = "_".join(map(str, feature))
        with open(os.path.join(save_dir, f'PINNs_results_{architecture_name}.json'), "w") as write_file:
            json.dump(results, write_file)
 
        with open(os.path.join(save_dir, f'PINNs_evaluation_{architecture_name}.json'), "w") as write_file:
            json.dump(evaluation, write_file)
        
        print(json.dumps(evaluation, indent=4))


if __name__ == '__main__':
    main()