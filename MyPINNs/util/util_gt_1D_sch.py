import jax.numpy as jnp
import time, jax, os, json
import scipy.io

class ImportData:
    def __init__(self,name_folder=None, save_dir = './Eval_Points/'):
        self.save_dir = save_dir + name_folder + '/'

    def get_FEM_coordinates(self):
        with open(os.path.join(self.save_dir,'eval_coordinates.json'), 'r') as f:
            eval_coordinates= json.load(f)
        mesh_coord = eval_coordinates['mesh_coord']['0']
        dt_coord = eval_coordinates['dt_coord']['0']
        return mesh_coord, dt_coord

    def get_FEM_results(self):
        with open(os.path.join(self.save_dir,'eval_solution_mat.json'), 'r') as f:
            eval_solution_mat= json.load(f)
        eval_solution_mat = jnp.asarray(eval_solution_mat)
        true_u = eval_solution_mat[0]
        true_v = eval_solution_mat[1]
        true_h = eval_solution_mat[2]
        return true_u, true_v, true_h

        
def get_relative_error(u,v):
        l2 = jnp.linalg.norm(u - v)/jnp.linalg.norm(u)
        return l2

class CompareGT:

    def get_FEM_comparison(mesh_coord,dt_coord,FEM_real,FEM_imag,FEM_sq,model,tuned_params): 
        dom_mesh = jnp.asarray(mesh_coord).squeeze()
        dom_mesh_ = jnp.tile(dom_mesh,len(dt_coord))#repeating the dom_mesh, len(dt_coord)-times
        dom_ts = jnp.repeat(jnp.array(dt_coord),len(mesh_coord))#repeating ts, len(mesh_coord)-times
        domain_pt = jnp.stack((dom_ts,dom_mesh_),axis=1) #stacking them together, meaning for each mesh coordinate we look at every time instance in ts
        
        tuned_params = jax.device_put(tuned_params)
        domain_pt = jax.device_put(domain_pt)

        start_time = time.time()
        approx = jax.block_until_ready(model(tuned_params, domain_pt, dim=2).squeeze()) 
        times_eval = time.time()-start_time
        
        approx = approx.reshape(len(dt_coord),len(mesh_coord),2)
        h_approx = jnp.sqrt(approx[:,:,0]**2+approx[:,:,1]**2) 
        real_l2 = []
        imag_l2 = []
        sq_l2 = []

        for l in range(len(dt_coord)): 
            real_l2.append(get_relative_error(FEM_real[int(l)],approx[int(l),:,0]))
            imag_l2.append(get_relative_error(FEM_imag[int(l)],approx[int(l),:,1]))
            sq_l2.append(get_relative_error(FEM_sq[int(l)],h_approx[int(l)]))

        return real_l2, imag_l2, sq_l2, times_eval, approx, h_approx, FEM_real, FEM_imag, FEM_sq, domain_pt
