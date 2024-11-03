from __future__ import print_function
from fenics import *
import numpy as np
import time
import json, os

########################################################################
# Solve PDE with FEM
########################################################################
tol = 1E-14
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

  # Left boundary is "target domain"
  def inside(self, x, on_boundary):
    return bool(x[0] < -5.0 + tol and x[0] > -5.0 -tol and on_boundary)

  def map(self, x, y):
    y[0] = x[0] - 10.0

# Create periodic boundary condition
pbc = PeriodicBoundary()

dt = 1e-4 # Size of the time step
T = np.pi / 2
num_steps = int(T/dt)
nums = [7993] # Mesh spacings that will be investigated, power of 2 here, maybe 2048

results, solution=dict({}),dict({}) # Save amplitudes, evaluation times, solution times, errors
all_times = [dt*(n+1) for n in range(int(num_steps))] # List of all times for which we get the solution, will be useful for evaluation
dt_coords_size = 100
indices = np.random.randint(len(all_times),size=dt_coords_size)
indices = np.sort(indices)
saved_times = np.array(all_times)[indices]
true_u = np.zeros((dt_coords_size,nums[0]+1))
true_v = np.zeros((dt_coords_size,nums[0]+1))
true_h = np.zeros((dt_coords_size,nums[0]+1))
sol_matrix = []
n_sol=0
eval_coordinates = {}
eval_coordinates['mesh_coord'] = {}
eval_coordinates['dt_coord'] = {}                          
eval_coordinates['dt_coord']['0'] = list(saved_times)  

av_iter_sol = 1 # Over how many iterations we want to average the solution time
for num in nums:
  print('Start solving', num)
  mesh = IntervalMesh(num,-5.0, 5.0) # Declare mesh
  eval_coordinates['mesh_coord']['0'] = mesh.coordinates().tolist()
  V = VectorFunctionSpace(mesh, 'CG', 1, dim = 2, constrained_domain = pbc) # Periodic BC are included in the definition of the function space
  # CG is type of the finite element, and 1 is the degree
  # Here vector space is used because we must write separate equations for real and imaginary parts of h, and h is [h_re , h_im]

  results[num], solution[num] = dict({}), dict({})
  time_solving = 0
  for i in range(av_iter_sol):
    print('Average iter sol: ', i)
    t=0
    u_0 = Expression(  ( ' 2*pow(cosh(x[0]), -1)', '0'), degree = 1) # Initial value. Has the real part only
    u_n = interpolate(u_0, V) # At t=0, the solution from the last iteration is just u_0. u_0 must be made into the function in the space V.

    filepath = str(num)+"iter_" + str(0) # Save the initial condition to load it in the evaluation stage
    hdf = HDF5File(MPI.comm_world, filepath, "w")
    hdf.write(u_n, "/f")  
    hdf.close()

    u = TrialFunction(V) # Declare u as the trial function. It is the technical requirement for solve(a==L)
    v = TestFunction(V)  # Declare test function
    
    # PDE in the weak form. Separated into Re and Im parts
    F_Re = (-u[1]+u_n[1])*v[0]*dx - 0.5 * dot(grad(u[0]),grad(v[0]))*dt*dx + (u_n[0]**2+u_n[1]**2)*u[0]*v[0]*dt*dx
    F_Im = (u[0]-u_n[0])*v[1]*dx - 0.5 * dot(grad(u[1]),grad(v[1]))*dt*dx + (u_n[0]**2+u_n[1]**2)*u[1]*v[1]*dt*dx

    a_Re, L_Re = lhs(F_Re) , rhs(F_Re) # Fenics is able to work out lhs and rhs terms on its own
    a_Im, L_Im = lhs(F_Im) , rhs(F_Im)
    a = a_Re + a_Im
    L = L_Re + L_Im
    u = Function(V) # Declare u as the function that stores the solution

    t0 = time.time()
    for n in range(1,int(num_steps)+1):
        print('Time step: ', n)
        # Update current time
        t += dt
        # Compute solution        
        solve(a == L, u, bcs = None, solver_parameters={'linear_solver':'gmres'})
        # Update previous solution
        u_n.assign(u)

        filepath = './1D-Schroedinger-FEM/Approx-Solution-semiimplicit/' + str(num)+"iter_" + str(n) 
        hdf = HDF5File(MPI.comm_world, filepath, "w")
        hdf.write(u, "/f")  
        hdf.close()
        
        if t in saved_times:
            solution_values = u.vector().get_local().reshape((-1, 2))
            true_u[n_sol,:] = solution_values[:,0]
            true_v[n_sol,:] = solution_values[:,1]
            true_h[n_sol,:] = np.sqrt(true_u[n_sol,:]**2 + true_v[n_sol,:]**2)
            n_sol += 1

    t1 = time.time()
    time_solving += t1 - t0
  
  sol_matrix.append(true_u)
  sol_matrix.append(true_v)
  sol_matrix.append(true_h)
  tot_solve = (time_solving) / av_iter_sol
  results[num]['time_solve'] = tot_solve

  save_dir = './Eval_Points/'
  save_path = os.path.join(save_dir,'1D_Schroedinger')
  if not os.path.exists(save_path):
    os.makedirs(folder_path)

  with open(os.path.join(save_path,'eval_coordinates.json'), "w") as write_file:
    json.dump(eval_coordinates, write_file)

  with open(os.path.join(save_path,'eval_solution_mat.json'), "w") as write_file:
    json.dump(sol_matrix, write_file)
