import os
#import optax
import jax, time, pickle
import jax.numpy as jnp
from jax import debug
import numpy as onp
from functools import partial
import json

import numpy.random as npr

from nn.model import ANN_emb, initialize_params, Embedder
from optimizers import Adam2 as Adam
from lr_schedulers import LinearWarmupCosineDecay
from dataset.util_Allen_1D import sample_points, sample_training_points, plot_losses

from util_gt import ImportData, CompareGT_embd
#from Allen_Cahn_1D.util import sample_points

embedder = Embedder(input_dims=1, include_input=True, max_freq_log2=4, num_freqs=6, log_sampling=True)

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# PDE residual for 1D Allen-Cahn
@partial(jax.vmap, in_axes=(None, 0, 0, None), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, t, x, eps=0.01):
    u_t = jax.jvp(u, (t, x), (1., 0.))[1]
    u_xx = jax.hessian(u,argnums=1)(t,x)
    return u_t - eps*u_xx  + (1/eps)*2*u(t,x)*(1-u(t,x))*(1-2*u(t,x))


# Inital condition
@partial(jax.vmap, in_axes=0)
def u_init(xs):
    return jnp.array([0.5*(0.5*jnp.sin(xs*2*jnp.pi) + 0.5*jnp.sin(xs*16*jnp.pi)) + 0.5])

# Loss functionals
@jax.jit
def pde_residual(params, points):
    eps = 0.01
    return jnp.mean(residual(lambda t, x: ANN_emb(params, jnp.stack((t, x)), embedder), points[:, 0], points[:, 1],eps)**2)
    
@partial(jax.jit, static_argnums=0)
def init_residual(u_init,params, xs):
    ini_approx = ANN_emb(params, jnp.stack((jnp.zeros_like(xs[:,0]), xs[:,0]), axis=1), embedder)
    ini_true = u_init(xs[:,0])
    return jnp.mean((ini_approx - ini_true)**2)
    
@jax.jit
def boundary_residual(params, ts): #periodic bc
    return jnp.mean((ANN_emb(params, jnp.stack((ts[:,0], jnp.zeros_like(ts[:,0])), axis=1), embedder) - ANN_emb(params, jnp.stack((ts[:,0], jnp.ones_like(ts[:,0])), axis=1), embedder))**2)
    
#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step_ini(params, opt, opt_state, val_points):
    init = sample_training_points([0.,0.],[0.05,1.],20000,250,500, val_points, init_only=True)

    init = jax.device_put(init)

    loss_val, grad = jax.value_and_grad(lambda params: init_residual(u_init,params, init))(params)    
    params, opt_state = opt.update(params, grad, opt_state)
    return params, opt_state, loss_val

@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, val_points):#, u_init):
    domain_points, boundary, init = sample_training_points([0.,0.],[0.05,1.],20000,250,500, val_points)

    domain_points = jax.device_put(domain_points)
    boundary = jax.device_put(boundary)
    init = jax.device_put(init)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    1000*init_residual(u_init,params, init) +
                                                    boundary_residual(params, boundary))(params)
    
    #pde_val, ini_val, bound_val = pde_residual(params, domain_points), init_residual(u_init,params, init), boundary_residual(params, boundary)
    params, opt_state = opt.update(params, grad, opt_state)
    return params, opt_state, loss_val

@jax.jit
def validation_step_ini(params, val_points_init):
    # Compute validation loss init
    loss_val = init_residual(u_init, params, val_points_init)
    return loss_val

@jax.jit
def validation_step(params, val_points):
    # Generate validation points
    val_domain_points, val_boundary, val_init = val_points
    
    # Compute validation loss
    loss_val = (
        pde_residual(params, val_domain_points) +
        1000 * init_residual(u_init, params, val_init) +
        boundary_residual(params, val_boundary)
    )

    return loss_val


def train_loop(params, adam, opt_state, init_epochs, num_epochs, val_points, n_patience, validate_every=10, lr_scheduler=None):
    train_losses = []
    val_losses = []
    best_loss = 3000
    patience = n_patience

    if init_epochs is not None:
        print('First stage: ')
        for init_epoch in range(init_epochs):
            params, opt_state, loss_train_init = training_step_ini(params, adam, opt_state, val_points)

            if (init_epoch + 1) % validate_every == 0:
                loss_val_init = validation_step_ini(params, val_points_init=val_points[-1])  # Compute validation loss
                #val_losses.append(loss_val.item())
                print(f"Epoch {init_epoch + 1}/{init_epochs} - Train Loss: {loss_train_init.item():.6f}, Val Loss: {loss_val_init.item():.6f}")
            #else:
            #    print(f"Init Epoch {epoch + 1}/{num_epochs} - Train Loss: {loss_train.item():.6f}")
        
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
            if best_loss - val_losses[-1] > 0.01:
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


#----------------------------------------------------
# Define Helper Functions for L-BFGS wrapper
#----------------------------------------------------
#def concat_params(params):
#    params, tree = jax.tree_util.tree_flatten(params)
#    shapes = [param.shape for param in params]
#    return jnp.concatenate([param.reshape(-1) for param in params]), tree, shapes

#def unconcat_params(params, tree, shapes):
#    split_vec = jnp.split(params, onp.cumsum([onp.prod(shape) for shape in shapes]))
#    split_vec = [vec.reshape(*shape) for vec, shape in zip(split_vec, shapes)]
#    return jax.tree_util.tree_unflatten(tree, split_vec)


def main():
    print('Device: ', jax.default_backend())

    lr = 1e-3
    init_epochs = 7000
    total_epochs = 50000
    validation_freq = 50

    val_domain_points, val_boundary, val_init = sample_points([0.,0.],[0.05,1.],2000,100,100)
    
    val_domain_points = jax.device_put(val_domain_points)
    val_boundary = jax.device_put(val_boundary)
    val_init = jax.device_put(val_init)

    #----------------------------------------------------
    # Define architectures list
    #----------------------------------------------------
    #architecture_list = [[2,20,20,20,1],[2,100,100,100,1],[2,500,500,500,1],[2,20,20,20,20,1],[2,100,100,100,100,1],[2,500,500,500,500,1],[2,20,20,20,20,20,1],[2,100,100,100,100,100,1],[2,500,500,500,500,500,1],[2,20,20,20,20,20,20,1],[2,100,100,100,100,100,100,1],[2,500,500,500,500,500,500,1],[2,20,20,20,20,20,20,20,1],[2,100,100,100,100,100,100,100,1]]
    architecture_list = [[14,20,20,20,1]]

    #----------------------------------------------------
    # Define embedder
    #----------------------------------------------------
    #embedder = Embedder(input_dims=1, include_input=True, max_freq_log2=4, num_freqs=6, log_sampling=True, periodic_fns=[jnp.sin, jnp.cos])

    #----------------------------------------------------
    # Load GT solution
    #----------------------------------------------------
    GTloader = ImportData(name_folder='1D_Allen-Cahn')
    mesh_coord, dt_coord = GTloader.get_FEM_coordinates()
    FEM = GTloader.get_FEM_results()

    #----------------------------------------------------
    # Train PINN
    #----------------------------------------------------
    # Train model 10 times and average over the times
    u_results = dict({})
    times_adam, times_eval, l2_rel, var, arch  = None, None, None, None, None #dict({}), dict({}), dict({}), dict({}), dict({})
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
            train_losses, val_losses, params, opt_state, = train_loop(params, optimizer, opt_state, init_epochs, total_epochs, n_patience=5, validate_every=validation_freq, lr_scheduler=lr_scheduler, val_points=[val_domain_points,val_boundary,val_init])
            adam_time = time.time()-start_time
            times_adam_temp.append(adam_time)
            #print("Adam training time: ", adam_time)

            #----------------------------------------------------
            # Start Training with L-BFGS optimiser
            #----------------------------------------------------
            #init_point, tree, shapes = concat_params(params)
            #domain_points, boundary, init = sample_points([0.,0.],[0.05,1.],20000,250,500)

            #print('Starting L-BFGS Optimisation')
            #start_time2 = time.time()
            #results = tfp.optimizer.lbfgs_minimize(jax.value_and_grad(lambda params: pde_residual(unconcat_params(params, tree, shapes), domain_points) + 
            #                                                    1000*init_residual(u_init,unconcat_params(params, tree, shapes), init) +
            #                                                    boundary_residual(unconcat_params(params, tree, shapes), boundary)),
            #                            init_point,
            #                            max_iterations=50000,
            #                            num_correction_pairs=50,
            #                            f_relative_tolerance=1.0 * jnp.finfo(float).eps)
        
            #lbfgs_time = time.time()-start_time2
            #times_total_temp.append(time.time()-start_time)
            #times_lbfgs_temp.append(lbfgs_time)

            # Evaluation
            #tuned_params = unconcat_params(results.position, tree, shapes)
            
            l2, times_temp, approx, gt_fem, domain_pt = CompareGT.get_FEM_comparison(mesh_coord,dt_coord,FEM,ANN_emb,params)
            times_eval_temp.append(times_temp)
            l2_errors.append(jnp.mean(jnp.array(l2)))

        u_gt = gt_fem.tolist()
        domain_pts = domain_pt.tolist()
        u_results = approx.tolist()
        times_adam, times_eval, l2_rel, var, arch = onp.mean(times_adam_temp), onp.mean(times_eval_temp), onp.mean(jnp.array(l2_errors)).tolist(), onp.var(jnp.array(l2_errors)).tolist(), feature

        results = dict({'domain_pts': domain_pts,
                        'u_results': u_results,
                        'u_gt': u_gt})

        evaluation = dict({'arch': arch,
            'times_adam': times_adam,
            'times_eval': times_eval,
            'l2_rel': l2_rel,
            'var_u': var})

        save_dir = './MyPINNS_results/1D-Allen-Cahn-PINN'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        architecture_name = "_".join(map(str, feature))
        with open(os.path.join(save_dir, f'PINNs_results_smalleps_{architecture_name}.json'), "w") as write_file:
            json.dump(results, write_file)
 
        with open(os.path.join(save_dir, f'PINNs_evaluation_smalleps_{architecture_name}.json'), "w") as write_file:
            json.dump(evaluation, write_file)
        
        print(json.dumps(evaluation, indent=4))


if __name__ == '__main__':
    main()