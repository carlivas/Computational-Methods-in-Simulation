import igl
import meshplot as mp
import numpy as np
import matplotlib.pyplot as plt

def make_beam_mesh(width, height, shape):
    x0 = -width/2.0
    y0 = -height/2.0
    I  = shape[0]
    J  = shape[1]
    dx = width/float(I)
    dy = height/float(J)
    V = np.zeros(((I+1)*(J+1),2),dtype=np.float64)
    for j in range(J+1):
        for i in range(I+1):
            k = i + j*(I+1)
            V[k,0] = x0 + i*dx
            V[k,1] = y0 + j*dy
    T = np.zeros((2*I*J,3),dtype=np.int32)
    for j in range(J):
        for i in range(I):
            k00 = (i  ) + (j  )*(I+1)
            k01 = (i+1) + (j  )*(I+1)
            k10 = (i  ) + (j+1)*(I+1)
            k11 = (i+1) + (j+1)*(I+1)
            e = 2*(i + j*I)
            if (i+j)%2:
                T[e,  :] = (k00,k01,k11)
                T[e+1,:] = (k00,k11,k10)
            else:
                T[e,  :] = (k10,k00,k01)
                T[e+1,:] = (k10,k01,k11)                    
    return V, T

def compute_triangle_areas(V,T):
    E = len(T) # Total number of triangles in the mesh
    A = np.zeros((E,),dtype=np.float64)
    for e in range(E):
        # Get triangle indices
        i = T[e,0]
        j = T[e,1]
        k = T[e,2]
        # Get triangle coordinates
        xi = V[i,0]
        xj = V[j,0]
        xk = V[k,0]
        yi = V[i,1]
        yj = V[j,1]
        yk = V[k,1]    
        
        dx1 = xk - xj
        dy1 = yk - yj
        dx2 = xi - xj
        dy2 = yi - yj

        A[e] =  (dx1*dy2 - dx2*dy1 ) / 2.0
    return A

def assemble_local_K_matrices(V, T):
    N = len(V) # Total number of nodes in the mesh
    E = len(T) # Total number of triangles in the mesh

    A = compute_triangle_areas(V,T)
    Ke = np.zeros((E, 3, 3),dtype=np.float64)

    # EMBARRASSINGLY PARALLEL
    for e in range(E):
        # Get triangle indices
        i = T[e,0]
        j = T[e,1]
        k = T[e,2]
        # Get triangle coordinates
        xi = V[i,0]; yi = V[i,1]
        xj = V[j,0]; yj = V[j,1]
        xk = V[k,0]; yk = V[k,1] 
        
        # TODO - Compute element matrix and store it in Ke array
        A_ijk = A[e]
        
        D_Ni_x = -(yk - yj) / (2 * A_ijk)
        D_Nj_x = -(yi - yk) / (2 * A_ijk)
        D_Nk_x = -(yj - yi) / (2 * A_ijk)
        
        D_Ni_y = (xk - xj) / (2 * A_ijk)
        D_Nj_y = (xi - xk) / (2 * A_ijk)
        D_Nk_y = (xj - xi) / (2 * A_ijk)
        
        D_Ne_x = np.array([D_Ni_x, D_Nj_x, D_Nk_x])
        D_Ne_y = np.array([D_Ni_y, D_Nj_y, D_Nk_y])
        
        element_K = A_ijk * (np.outer(D_Ne_x, D_Ne_x) + np.outer(D_Ne_y, D_Ne_y))
        Ke[e] = element_K
    return Ke
    
def assemble_global_K_matrix(V, T, Ke):
    N = len(V) # Total number of nodes in the mesh
    E = len(T) # Total number of triangles in the mesh
    
    K  = np.zeros((N,N),dtype=np.float64) 
    for e in range(E):
        # Get global triangle vertex indices
        i = T[e,0]
        j = T[e,1]
        k = T[e,2]
        # Local order of vertex coordinates is i, j and k. 
        # This is how local vertex indices (0,1,2) are mapped to global vertex
        # indices
        gidx = [i, j, k]
        # TODO - do assembly of Ke into K
        for p in range(3):
            for q in range(3):
                K[gidx[p], gidx[q]] += Ke[e,p,q]
    return K

def apply_boundary_conditions(V, K, left_boundary_func, right_boundary_func, width, height, I, J, return_plots=False, verbose=False):
    N = len(V) # Total number of nodes in the mesh
    dx = width / I
    dy = height / J
    left_boundary_points = np.array( np.where(V[:,0] < -width/2*(1 - 1/I)), dtype=np.int32).flatten()
    right_boundary_points = np.array( np.where(V[:,0] > width/2*(1 - 1/I)), dtype=np.int32).flatten()
    left_boundary_values = left_boundary_func(V[left_boundary_points,1])
    right_boundary_values = right_boundary_func(V[right_boundary_points,1])
    boundary_indices = np.hstack((left_boundary_points, right_boundary_points))
    boundary_values  = np.hstack((left_boundary_values, right_boundary_values))

    F = np.setdiff1d(np.arange(N), boundary_indices)
    f  = np.zeros((N,),dtype=np.float64)

    for i, (boundary_index, boundary_value) in enumerate(zip(boundary_indices, boundary_values)):
        # TODO - Insert boundary conditions into K
        
        # Moving known values to the right_boundary_points hand side
        f[F] -= K[F, boundary_index] * boundary_value
        f[boundary_index] = boundary_value

        # Inserting boundary conditions into K
        K[boundary_index,:] = 0
        K[:, boundary_index] = 0
        K[boundary_index,boundary_index] = 1
    
    # Initializing y with zeros and boundary values
    y = np.zeros(f.shape, dtype=np.float64)
    y[boundary_indices] = boundary_values
    
    if return_plots:
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.spy(K)
        ax.set_title('Fill pattern of matrix after boundary conditions')
        ax.set_ylabel('Row index')
        ax.set_xlabel('Column index')
        plt.show()
        
        d, _ = np.linalg.eig(K)
        d_sort = np.sort(d)
        fig = plt.figure()
        plt.plot(d_sort, '.' )
        ax.set_title('Eigenvalues of matrix')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Value')
        plt.show()
        
    if verbose:    
        print(f"The linear system has {(d.any() <= 0).sum()} eigenvalues with values of zero or less. ")
    return K, f, y, F

def solve_heat_eq(width, height, I, J, left_boundary_condition, right_boundary_condition, return_plots=False, verbose=False):
    V, T = make_beam_mesh(width,height,(I,J))
    Ke = assemble_local_K_matrices(V, T)
    K = assemble_global_K_matrix(V, T, Ke)
    K, f, y, F = apply_boundary_conditions(V, K, left_boundary_condition, right_boundary_condition, width, height, I, J, return_plots, verbose)
    
    # Solving the linear system inside the domain
    KFF = K[F,:][:,F]
    fF  = f[F]
    y[F] = np.linalg.solve(KFF, fF)
    
    if return_plots:
        N = len(V)
        xs = np.linspace(-width/2, width/2, N)
        ys = np.linspace(-height/2, height/2, N)
            
        X, Y = np.meshgrid(xs, ys)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(V[:,0],V[:,1],y, cmap='viridis')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("phi")
        plt.show()
    return y, V, T