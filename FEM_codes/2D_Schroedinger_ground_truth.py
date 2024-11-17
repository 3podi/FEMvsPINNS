from __future__ import print_function
from fenics import *
import numpy as np
import time, json, os

########################################################################
# Solve PDE with FEM
########################################################################
#Definitions of mesh functions, loading eval points etc
tol = 1E-14
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], -5) or near(x[1], -5)) and 
                (not ((near(x[0], -5) and near(x[1], 5)) or 
                        (near(x[0], 5) and near(x[1], -5)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1): # Top right corner
            y[0] = x[0] - 10.
            y[1] = x[1] - 10.
        elif near(x[0], 1): # Right edge
            y[0] = x[0] - 10.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 10.

# Create periodic boundary condition
pbc = PeriodicBoundary()

dt = 1e-4
T = np.pi/2
num_steps = int(T/dt)
nums = [(258,258)]
av_iter_sol = 1 # Over how many iterations we want to average

dt_coords_size = 100  
sol_matrix = []
n_sol=0
eval_coordinates = {}
eval_coordinates['mesh_coord'] = {}
eval_coordinates['dt_coord'] = {}

results, solution = dict({}),dict({}) # Save eval times, solution times, errors

for num in nums:
  numx = num[0]
  numy = num[1]
  print('Start solving', numx)
  mesh = RectangleMesh(Point(-5,-5), Point(5, 5), numx, numy)
  eval_coordinates['mesh_coord']['0'] = mesh.coordinates().tolist()
  V = VectorFunctionSpace(mesh, 'CG', 1, dim = 2, constrained_domain = pbc) # periodic BC are included in the definition of the function space
  # Here vector space is used because we must write separate equations for real and imaginary parts of h, and h is [h_re , h_im]

  all_times = [dt*(n+1) for n in range(int(num_steps))] # List of all times for which we get the solution, will be useful for evaluation. We do not start from t = 0
  results[numx] = dict({})
  indices = np.random.choice(range(len(all_times)), size=dt_coords_size, replace=False)
  indices = np.sort(indices) + 1
  print('Timesteps saved for GT: ', indices)
  saved_times = np.array(all_times)[indices-1]
  eval_coordinates['dt_coord']['0'] = list(saved_times)
  true_u = np.zeros((dt_coords_size,len(eval_coordinates['mesh_coord']['0'])))
  true_v = np.zeros((dt_coords_size,len(eval_coordinates['mesh_coord']['0'])))
  true_h = np.zeros((dt_coords_size,len(eval_coordinates['mesh_coord']['0'])))
  
  for i in range(av_iter_sol):
    t=0
    time_solving = 0
    u_0 = Expression(  ( 'pow(cosh(x[0]), -1) + 0.5 * ( pow(cosh(x[1]-2), -1) + pow(cosh(x[1]+2), -1) )', '0'), degree = 1) # Initial value. Has the real part only
    u_n = interpolate(u_0, V)
    u = TrialFunction(V)

    v = TestFunction(V)
    F_Re = (-u[1]+u_n[1])*v[0]*dx - 0.5 * dot(grad(u[0]),grad(v[0]))*dt*dx + (u_n[0]**2+u_n[1]**2)*u[0]*v[0]*dt*dx # 
    F_Im = (u[0]-u_n[0])*v[1]*dx - 0.5 * dot(grad(u[1]),grad(v[1]))*dt*dx + (u_n[0]**2+u_n[1]**2)*u[1]*v[1]*dt*dx

    a_Re, L_Re = lhs(F_Re) , rhs(F_Re)
    a_Im, L_Im = lhs(F_Im) , rhs(F_Im)
    a = a_Re + a_Im
    L = L_Re + L_Im
    u = Function(V)

    save_dir = os.path.join('./2D-Schroedinger-FEM/Approx-Solution-semiimplicit/','Mesh_%03d' %numx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t0 = time.time()
    for n in range(int(num_steps)):
        # Update current time
        t = dt * n
        # Compute solution        
        solve(a == L, u, bcs = None, solver_parameters={'linear_solver':'gmres'}) 
        # Update previous solution
        u_n.assign(u)

        filepath = os.path.join(save_dir,'iter_%05d' %n)
        hdf = HDF5File(MPI.comm_world, filepath, "w")
        hdf.write(u, "/f")  
        hdf.close()
       
        if n in indices:
          print(f'Saving timestep {n_sol+1}/{dt_coords_size}')
          solutions_at_eval_points = []

          # Loop through each evaluation point
          for point in eval_coordinates['mesh_coord']['0']:
            x_eval, y_eval = point
            # Interpolate the solution at the evaluation point
            u_eval = u(x_eval, y_eval)  # Directly access the function value at (x, y)

            # Store the results (real and imaginary parts)
            solutions_at_eval_points.append((u_eval[0], u_eval[1]))  # (Real, Imaginary)

          # Convert to a NumPy array for easier handling
          true_u[n_sol, :] = [sol[0] for sol in solutions_at_eval_points]
          true_v[n_sol, :] = [sol[1] for sol in solutions_at_eval_points]
          true_h[n_sol, :] = np.sqrt(true_u[n_sol, :]**2 + true_v[n_sol, :]**2)
          n_sol += 1

    t1 = time.time()
    time_solving += t1 - t0

  sol_matrix = [true_u.tolist(), true_v.tolist(), true_h.tolist()]
  tot_solve = (time_solving) / av_iter_sol
  results[numx]['time_solve'] = tot_solve
    
  save_dir = './Eval_Points/'
  save_path = os.path.join(save_dir,'2D_Schroedinger')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
            
  with open(os.path.join(save_path,'eval_coordinates.json'), "w") as write_file:
    json.dump(eval_coordinates, write_file)
  
  with open(os.path.join(save_path,'eval_solution_mat.json'), "w") as write_file:
    json.dump(sol_matrix, write_file)
