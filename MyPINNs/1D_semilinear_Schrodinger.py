import os
import jax, time
import jax.numpy as jnp
import numpy as onp
from functools import partial
import json

from nn.model import initialize_params
from nn.model import ANN2 as ANN
from optimizers import Adam2 as Adam
from lr_schedulers import LinearWarmupCosineDecay
from dataset.util_Sch_1D import sample_points, sample_training_points

from 1D_Sch_semilinear.util_gt import ImportData, CompareGT 

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# PDE residual for 1D Semilinear Schrödinger
@partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, t, x):
    u_t = jax.jvp(u, (t, x), (1., 0.))[1]
    u_real_t = u_t[0]
    u_imag_t = u_t[1]
    u_xx = jax.hessian(u, argnums=1)(t, x)
    u_real_xx = u_xx[0]
    u_imag_xx = u_xx[1]

    h = (u(t,x)[0])**2 + (u(t,x)[1])**2

    f_real = -u_imag_t + 0.5*u_real_xx + h*u(t,x)[0] 
    f_imag = u_real_t + 0.5*u_imag_xx + h*u(t,x)[1]
    return f_real, f_imag

# Inital condition
@partial(jax.vmap, in_axes=0)
def u_init(xs):
    return jnp.array([2./jnp.cosh(xs), 0.])

# Loss functionals
@jax.jit
def pde_residual(params, points):
    f_real, f_imag = residual(lambda t, x: ANN(params, jnp.stack((t, x)),dim=2), points[:, 0], points[:, 1])
    return jnp.mean(f_real**2) + jnp.mean(f_imag**2)

@partial(jax.jit, static_argnums=0)
def init_residual(u_init, params, xs):
    return jnp.mean((ANN(params, jnp.stack((jnp.zeros_like(xs[:,0]), xs[:,0]), axis=1), dim=2) - u_init(xs[:,0]))**2)

@jax.jit
def boundary_residual(params, ts):
    return jnp.mean((ANN(params, jnp.stack((ts[:,0], 5 * jnp.ones_like(ts[:,0])), axis=1), dim=2) - 
                                  ANN(params, jnp.stack((ts[:,0], -5 * jnp.ones_like(ts[:,0])), axis=1), dim=2))**2)

#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, val_points):#, u_init):
    domain_points, boundary, init = sample_training_points([0.,-5.],[1.,5.],20000,50,50, val_points)

    domain_points = jax.device_put(domain_points)
    boundary = jax.device_put(boundary)
    init = jax.device_put(init)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                       init_residual(u_init,params, init) +
                                                       boundary_residual(params, boundary))(params)
    
    params, opt_state = opt.update(params, grad, opt_state)
    return params, opt_state, loss_val


@jax.jit
def validation_step(params, val_points):
    # Generate validation points
    val_domain_points, val_boundary, val_init = val_points
    
    # Compute validation loss
    loss_val = (
        pde_residual(params, val_domain_points) +
        init_residual(u_init, params, val_init) +
        boundary_residual(params, val_boundary)
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
            
            #if patience == 0:
            #    print('Early stopping the training, best val_loss: ', best_loss)
            #    break
            
    
    return train_losses, val_losses, params, opt_state


def main():
    print('Device: ', jax.default_backend())

    lr = 1e-3
    total_epochs = 50000
    validation_freq = 50

    val_domain_points, val_boundary, val_init = sample_points([0.,-5.],[1.,5.],5000,100,100)
    
    val_domain_points = jax.device_put(val_domain_points)
    val_boundary = jax.device_put(val_boundary)
    val_init = jax.device_put(val_init)

    #----------------------------------------------------
    # Define architectures list
    #----------------------------------------------------
    architecture_list = [[2,20,20,20,2],[2,100,100,100,2],[2,20,20,20,20,2],[2,100,100,100,100,2],[2,20,20,20,20,20,2],[2,100,100,100,100,100,2],[2,20,20,20,20,20,20,2],[2,100,100,100,100,100,100,2]]

    #----------------------------------------------------
    # Load GT solution
    #----------------------------------------------------
    GTloader = ImportData(name_folder='1D_Schroedinger')
    mesh_coord, dt_coord = GTloader.get_FEM_coordinates()
    FEM = GTloader.get_FEM_results()

    #----------------------------------------------------
    # Train PINN
    #----------------------------------------------------
    # Train model 10 times and average over the times
    u_results = None
    v_results = None
    h_results = None
    times_adam, times_eval, l2_rel, var, arch  = None, None, None, None, None
    print('Start training')
    for feature in architecture_list:
        print('Architecture: ', feature)
        times_adam_temp = []
        times_eval_temp = []
        
        l2_errors_u = []
        l2_errors_v = []
        l2_errors_h = []

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
            train_losses, val_losses, params, opt_state, = train_loop(params, optimizer, opt_state, total_epochs, n_patience=5, validate_every=validation_freq, lr_scheduler=lr_scheduler, val_points=[val_domain_points,val_boundary,val_init])
            adam_time = time.time()-start_time
            times_adam_temp.append(adam_time)
            #print("Adam training time: ", adam_time)

            
            # Evaluation
            real_l2, imag_l2, sq_l2, times_temp, approx, h_approx, true_u, true_v, true_h, domain_pt = CompareGT.get_FEM_comparison(mesh_coord,dt_coord,FEM_real,FEM_imag,FEM_sq,model,tuned_params) #dt_coord_100,
            times_eval_temp.append(times_temp)
            l2_errors_u.append(jnp.mean(jnp.array(real_l2)))#real_l2)
            l2_errors_v.append(jnp.mean(jnp.array(imag_l2)))#imag_l2)
            l2_errors_h.append(jnp.mean(jnp.array(sq_l2)))#sq_l2)

        u_gt, v_gt, h_gt, domain_pts = true_u.tolist(), true_v.tolist(), true_h.tolist(), domain_pt.tolist()
        u_results[n], v_results[n], h_results[n] = approx[:,:,0].tolist(), approx[:,:,1].tolist(), h_approx.tolist()
        times_adam, times_eval, l2_rel_u, l2_rel_v, l2_rel_h, var_u, var_v, var_h, arch = onp.mean(times_adam_temp), onp.mean(times_eval_temp), onp.mean(jnp.array(l2_errors_u)).tolist(), onp.mean(jnp.array(l2_errors_v)).tolist(), onp.mean(jnp.array(l2_errors_h)).tolist(), onp.var(jnp.array(l2_errors_u)).tolist(), onp.var(jnp.array(l2_errors_v)).tolist(), onp.var(jnp.array(l2_errors_h)).tolist() , feature

        results = dict({'domain_pts': domain_pts,
                        'u_results': u_results,
                        'v_results': v_results,
                        'h_results': h_results,
                        'u_gt': u_gt,
                        'v_gt': v_gt,
                        'h_gt': h_gt})
        
        evaluation = dict({'arch': arch,
            'times_adam': times_adam,
            'times_eval': times_eval,
            'l2_rel_u': l2_rel_u,
            'l2_rel_v': l2_rel_v,
            'l2_rel_h': l2_rel_h,
            'var_u': var_u,
            'var_v': var_v,
            'var_h': var_h})

        save_dir = './MyPINNS_results/1D-Schroedinger-PINN'
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