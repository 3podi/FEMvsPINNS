import os
import jax, time
import jax.numpy as jnp
import numpy as onp
import scipy as sp
from functools import partial
import json

from nn.model import initialize_params
from nn.model import ANN2 as ANN
from optimizers import Adam2 as Adam
from lr_schedulers import LinearWarmupCosineDecay
from dataset.util_Sch_2D import sample_points, sample_training_points

from util.util_gt_2D_sch import ImportData, CompareGT 

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# PDE residual for 2D Semilinear Schrödinger
@partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, t, x, y):
    u_t = jax.jvp(u, (t, x, y), (1., 0., 0.))[1]
    u_real_t = u_t[0]
    u_imag_t = u_t[1]

    du_dxx = jax.hessian(u, argnums=1)(t,x,y)
    du_dyy = jax.hessian(u, argnums=2)(t,x,y)
    u_real_xx = du_dxx[0]
    u_imag_xx = du_dxx[1]
    u_real_yy = du_dyy[0]
    u_imag_yy = du_dyy[1]

    h = (u(t,x,y)[0])**2 + (u(t,x,y)[1])**2

    f_real = -u_imag_t + 0.5*(u_real_xx+u_real_yy) + h*u(t,x,y)[0] 
    f_imag = u_real_t + 0.5*(u_imag_xx+u_imag_yy) + h*u(t,x,y)[1]

    return f_real, f_imag

# Initial condition
@jax.vmap
def u_init(xs,ys):
    return jnp.array([1./jnp.cosh(xs) + 0.5*(1./jnp.cosh(ys-2)) + 0.5*(1./jnp.cosh(ys+2)), 0.])

# Helper functions for boundary condition
@partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
def u_x(u,t,x,y):
    u_x = jax.jvp(u, (t,x,y), (0., 1., 0.))[1]
    return u_x

@partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
def u_y(u,t,x,y):
    u_y = jax.jvp(u, (t,x,y), (0., 0., 1.))[1]
    return u_y

# Loss functionals
@jax.jit
def pde_residual(params, points):
    f_real, f_imag = residual(lambda t, x, y: ANN(params, jnp.stack((t, x, y)),dim=3), points[:, 0], points[:, 1], points[:, 2])
    return jnp.mean(f_real**2) + jnp.mean(f_imag**2)

@partial(jax.jit, static_argnums=0)
def init_residual(u_init, params, points):
    return jnp.mean((ANN(params, jnp.stack((jnp.zeros(*points[:,0].shape), points[:,0], points[:,2]), axis=1), dim=3) - u_init(points[:,0],  points[:,2]))**2)

@jax.jit
def boundary_residual_x(params, points):
    return jnp.mean((ANN(params, jnp.stack((points[:,0], 5 * jnp.ones(*points[:,0].shape), points[:,2]), axis=1), dim=3) - 
                                  ANN(params, jnp.stack((points[:,0], -5 * jnp.ones(*points[:,0].shape), points[:,2]), axis=1), dim=3))**2)

@jax.jit
def boundary_residual_y(params, points):
    return jnp.mean((ANN(params, jnp.stack((points[:,0], points[:,1], 5 * jnp.ones(*points[:,0].shape)), axis=1), dim=3) - 
                                  ANN(params, jnp.stack((points[:,0], points[:,1], -5 * jnp.ones(*points[:,0].shape)), axis=1), dim=3))**2)

@jax.jit
def boundary_residual_x_der(params, points):
    u_x5 = u_x(lambda t, x, y: ANN(params, jnp.stack((t, x, y)), dim=3), points[:,0], 5 * jnp.ones(*points[:,0].shape), points[:,2])
    u_xm5 = u_x(lambda t, x, y: ANN(params, jnp.stack((t, x, y)), dim=3), points[:,0], -5 * jnp.ones(*points[:,0].shape), points[:,2])
    return jnp.mean((u_x5 - u_xm5)**2)

@jax.jit
def boundary_residual_y_der(params, points):
    u_y5 = u_y(lambda t, x, y: ANN(params, jnp.stack((t, x, y)), dim=3), points[:,0], points[:,1], 5 * jnp.ones(*points[:,0].shape))
    u_ym5 = u_y(lambda t, x, y: ANN(params, jnp.stack((t, x, y)), dim=3), points[:,0], points[:,1], -5 * jnp.ones(*points[:,0].shape))
    return jnp.mean((u_y5 - u_ym5)**2)

#----------
# Define L-BFGS
# ----------
# Flatten the parameters into a single vector for optimization
def flatten_params(params):
    return jnp.concatenate([param.flatten() for param in params])

def unflatten_params(flat_params, param_shapes):
    params = []
    start_idx = 0
    for shape in param_shapes:
        size = jnp.prod(shape)
        param = flat_params[start_idx:start_idx + size].reshape(shape)
        params.append(param)
        start_idx += size
    return params


# Loss and gradient function (we modify this to handle flattened parameters)
def compute_loss_and_grad(flat_params, domain_points, u_init, boundary, init, param_shapes):
    # Unflatten parameters
    params = unflatten_params(flat_params, param_shapes)
    
    # Compute the loss and gradients
    loss_val, grad = jax.value_and_grad(lambda params: 
                                         pde_residual(params, domain_points) + 
                                         init_residual(u_init, params, init) +
                                         boundary_residual_x(params, boundary) +
                                         boundary_residual_y(params, boundary) +
                                         boundary_residual_x_der(params, boundary) +
                                         boundary_residual_y_der(params,boundary))(params)
    
    # Flatten the gradient to match the optimization format
    grad_flat = flatten_params(grad)
    return loss_val, grad_flat

# L-BFGS Optimization Loop
def lbfgs_optimizer(ANN_params, domain_points=None, u_init=None, boundary=None, init=None, val_points=None, max_epochs=1000, tol=1e-8, m=10):

    # Initialize parameters
    param_shapes = [param.shape for param in ANN_params]  # Store original shapes for unflattening
    flat_params = flatten_params(ANN_params)  # Flatten the parameters
    epoch = 0
    history = []

    # Initialize L-BFGS variables
    s_history = []  # History of s = x_new - x
    y_history = []  # History of y = grad_new - grad
    rho_history = []  # History of rho = 1 / (y.T @ s)
    
    while epoch < max_epochs:
        print(epoch)
        epoch += 1
        
        domain_points, boundary, init = sample_training_points([0.,-5.,-5.],[1.,5.,5.],5000,100,100, val_points)

        domain_points = jax.device_put(domain_points)
        boundary = jax.device_put(boundary)
        init = jax.device_put(init)
        # Compute gradient and loss using the flattened parameters
        loss_val, grad = compute_loss_and_grad(flat_params, domain_points, u_init, boundary, init, param_shapes)

        # Check convergence
        if jnp.linalg.norm(grad) < tol:
            break
        
        # Compute search direction using L-BFGS approximation
        p = grad
        if len(s_history) > 0:
            for i in range(min(len(s_history), m) - 1, -1, -1):
                s_i = s_history[i]
                y_i = y_history[i]
                rho_i = rho_history[i]
                alpha = rho_i * jnp.dot(s_i, p)
                p -= alpha * y_i
            
            # Approximate Hessian is diagonal initially
            p *= 1.0  # This is a simple approach for diagonal Hessian approximation

            for i in range(min(len(s_history), m)):
                s_i = s_history[i]
                y_i = y_history[i]
                rho_i = rho_history[i]
                beta = rho_i * jnp.dot(y_i, p)
                p += (alpha - beta) * s_i

        # Line search (we use scipy's line search method)
        line_search = sp.optimize.line_search(lambda params: compute_loss_and_grad(params, domain_points, u_init, boundary, init, param_shapes)[0], 
                                              lambda params: compute_loss_and_grad(params, domain_points, u_init, boundary, init, param_shapes)[1], 
                                              flat_params, -p)
        alpha = line_search[0]  # Step size
        flat_params_new = flat_params - alpha * p  # Update parameters
        
        # Compute s and y
        s = flat_params_new - flat_params
        flat_params = flat_params_new
        loss_val_new, grad_new = compute_loss_and_grad(flat_params, domain_points, u_init, boundary, init, param_shapes)
        grad_new = grad_new.flatten()
        y = grad_new - grad

        # Update L-BFGS history
        if len(s_history) == m:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)
        s_history.append(s)
        y_history.append(y)
        rho_history.append(1.0 / jnp.dot(y, s))

        # Record the loss for history
        history.append(loss_val)

    # Unflatten the parameters back to their list form
    optimized_params = unflatten_params(flat_params, param_shapes)

    return optimized_params, history, epoch


#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, val_points):#, u_init):
    domain_points, boundary, init = sample_training_points([0.,-5.,-5.],[1.,5.,5.],5000,100,100, val_points)

    domain_points = jax.device_put(domain_points)
    boundary = jax.device_put(boundary)
    init = jax.device_put(init)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    init_residual(u_init, params, init) +
                                                    boundary_residual_x(params, boundary) +
                                                    boundary_residual_y(params, boundary)+
                                                    boundary_residual_x_der(params, boundary)+
                                                    boundary_residual_y_der(params,boundary))(params)
    
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
        boundary_residual_x(params, val_boundary) +
        boundary_residual_y(params, val_boundary)+
        boundary_residual_x_der(params, val_boundary)+
        boundary_residual_y_der(params, val_boundary)
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
            
    
    return train_losses, val_losses, params, opt_state


def main():
    print('Device: ', jax.default_backend())

    lr = 1e-3
    total_epochs = 50000
    validation_freq = 50

    val_domain_points, val_boundary, val_init = sample_points([0.,-5.,-5.],[1.,5.,5.],5000,100,100)
    
    val_domain_points = jax.device_put(val_domain_points)
    val_boundary = jax.device_put(val_boundary)
    val_init = jax.device_put(val_init)

    #----------------------------------------------------
    # Define architectures list
    #----------------------------------------------------
    architecture_list = [[3,20,20,20,2],[3,100,100,100,2],[3,20,20,20,20,2],[3,100,100,100,100,2],[3,20,20,20,20,20,2],[3,100,100,100,100,100,2]]

    #----------------------------------------------------
    # Load GT solution
    #----------------------------------------------------
    GTloader = ImportData(name_folder='2D_Schroedinger')
    mesh_coord, dt_coord = GTloader.get_FEM_coordinates()
    FEM_real,FEM_imag,FEM_sq = GTloader.get_FEM_results()

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
            train_losses, val_losses, params, opt_state, = train_loop(params, optimizer, opt_state, total_epochs, n_patience=10, validate_every=validation_freq, lr_scheduler=lr_scheduler, val_points=[val_domain_points,val_boundary,val_init])
            adam_time = time.time()-start_time
            times_adam_temp.append(adam_time)
            #print("Adam training time: ", adam_time)
            
            params, _, _ = lbfgs_optimizer(params, val_points=[val_domain_points,val_boundary,val_init], max_epochs=1000, tol=1e-8, m=10)

            # Evaluation
            real_l2, imag_l2, sq_l2, times_temp, approx, h_approx, true_u, true_v, true_h, domain_pt = CompareGT.get_FEM_comparison(mesh_coord,dt_coord,FEM_real,FEM_imag,FEM_sq,ANN,params) #dt_coord_100,
            times_eval_temp.append(times_temp)
            l2_errors_u.append(jnp.mean(jnp.array(real_l2)))#real_l2)
            l2_errors_v.append(jnp.mean(jnp.array(imag_l2)))#imag_l2)
            l2_errors_h.append(jnp.mean(jnp.array(sq_l2)))#sq_l2)

        u_gt, v_gt, h_gt, domain_pts = true_u.tolist(), true_v.tolist(), true_h.tolist(), domain_pt.tolist()
        u_results, v_results, h_results = approx[:,:,0].tolist(), approx[:,:,1].tolist(), h_approx.tolist()
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

        save_dir = './MyPINNS_results/2D-Schroedinger-PINN'
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