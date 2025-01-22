import os
import jax, time, pickle
import jax.numpy as jnp
from jax import debug
import numpy as onp
from functools import partial
import json

import numpy.random as npr

from nn.model import ANN, initialize_params
from optimizers import Adam2 as Adam
from lr_schedulers import LinearWarmupCosineDecay
from dataset.util_Poisson_1D import sample_points, sample_training_points

from util_gt import ImportData, CompareGT
#from Allen_Cahn_1D.util import sample_points


#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# Hessian-vector product
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(lambda x: f(x)[0]), primals, tangents)[1]

# PDE residual for 1D Poisson
@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x):
    v = jax.numpy.ones(x.shape)
    lhs = hvp(u,(x,),(v,))
    rhs = (-6*x + 4*x**3)*jax.numpy.exp(-x**2)
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x: ANN(params, x), points)**2)

@jax.jit
def boundary_residual0(params, xs):
    return jnp.mean((ANN(params, jnp.zeros_like(xs)))**2)

@jax.jit
def boundary_residual1(params, xs):
    return jnp.mean((ANN(params, jnp.ones_like(xs))-jnp.exp(-1.))**2) 
#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, val_points):#, u_init):
    domain_xs, boundary_xs = sample_training_points(val_points)

    domain_xs = jax.device_put(domain_xs)
    boundary_xs = jax.device_put(boundary_xs)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_xs) + 
                                                    boundary_residual0(params,boundary_xs) +
                                                    boundary_residual1(params, boundary_xs))(params)
    
    params, opt_state = opt.update(params, grad, opt_state)
    return params, opt_state, loss_val

@jax.jit
def validation_step(params, val_points):
    # Generate validation points
    val_domain_xs, val_boundary_xs = val_points
    
    # Compute validation loss
    loss_val = (pde_residual(params, val_domain_xs) + 
                boundary_residual0(params,val_boundary_xs) +
                boundary_residual1(params,val_boundary_xs)
                )

    return loss_val


def train_loop(params, adam, opt_state, num_epochs, val_points, n_patience, validate_every=10, lr_scheduler=None):
    train_losses = []
    val_losses = []
    best_loss = 3000
    patience = n_patience
        
    for epoch in range(num_epochs):
        # Lr scheduler step
        lr = lr_scheduler.get_lr()
        adam.learning_rate = lr

        # Perform a training step
        params, opt_state, loss_train = training_step(params, adam, opt_state, val_points)
        train_losses.append(loss_train.item())
        
        # Validation step (every `validate_every` epochs)
        if (epoch + 1) % validate_every == 0:
            loss_val = validation_step(params, val_points)  # Compute validation loss
            val_losses.append(loss_val.item())
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
            if best_loss - val_losses[-1] > 0.1:
                best_loss = val_losses[-1]  # Update best loss
                patience = n_patience  # Reset patience
            else:
                patience -= 1
            
            if patience == 0:
                print('Early stopping the training, best val_loss: ', best_loss)
                break
            
            #plot_losses(train_losses_dict, val_losses_dict)
        #else:
        #    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {loss_train.item():.6f}")
    
    return train_losses, val_losses, params, opt_state

def main():
    print('Device: ', jax.default_backend())

    lr = 1e-3
    total_epochs = 15000
    validation_freq = 50

    val_domain_points, val_boundary = sample_points()
    
    val_domain_points = jax.device_put(val_domain_points)
    val_boundary = jax.device_put(val_boundary)

    #----------------------------------------------------
    # Define architectures list
    #----------------------------------------------------
    architecture_list = [[1,1],[2,1],[5,1],[10,1],[20,1],[40,1],[5,5,1],[10,10,1],[20,20,1],[40,40,1],[5,5,5,1],[10,10,10,1],[20,20,20,1],[40,40,40,1]]
    
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
        accuracy_temp = []
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

            
            with open("./Eval_Points/1D_Poisson_eval-points.json", 'r') as f:
                domain_points = json.load(f)
                domain_points = jnp.array(domain_points)

            start_time3 = time.time()
            u_approx = ANN(params, domain_points).squeeze()
            times_eval_temp.append(time.time()-start_time3)

            u_true = (domain_points*jnp.exp(-domain_points**2)).squeeze()
            run_accuracy = (onp.linalg.norm(u_approx - u_true))/onp.linalg.norm(u_true)
            l2_errors.append(run_accuracy)

        y_gt = (domain_points*jnp.exp(-domain_points**2)).tolist()
        domain_pts = domain_pts.tolist()
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

        save_dir = './MyPINNS_results/1D-Poisson-PINN'
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