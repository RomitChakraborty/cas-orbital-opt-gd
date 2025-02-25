from pyscf import gto, scf, mcscf
import numpy as np
from jax.scipy.linalg import eigh
from pyscf.tools import cubegen
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.linalg import svd


def overlap_matrix_svd(C_CASSCF, C_trial, S):
    """Compute the singular values of the overlap matrix."""
    M = C_CASSCF.T.conj() @ S @ C_trial
    _, sigma, _ = np.linalg.svd(M)
    return sigma

def frobenius_norm_sines(sigma):
    """Compute the Frobenius norm of the matrix of sines of the principal angles."""
    sum_sines_squared = np.sum((1 - sigma**2))
    if (sum_sines_squared < 0 and sum_sines_squared > -1e-5):
        sum_sines_squared = 0
    return np.sqrt(sum_sines_squared)

# Use JAX versions of numpy and scipy operations
def overlap_matrix_svd_jax(C_CASSCF, C_trial, S):
    """
    Compute the singular values of the overlap matrix using JAX operations.
    """
    M = jnp.matmul(C_CASSCF.T.conj(), jnp.matmul(S, C_trial))   
    assert M.ndim == 2, "Matrix M must be 2-dimensional"
    U, sigma, VT = svd(M, full_matrices=False)  # SVD operation compatible with JAX for autodiff
    return sigma

def frobenius_norm_sines_jax(sigma):
    """
    Compute the Frobenius norm of the matrix of sines of the principal angles using JAX operations.
    """
    corrected_sigma = jnp.clip(sigma, 0, 1)  # Clip sigma values to [0, 1] range
    sum_sines_squared = jnp.sum((1 - corrected_sigma**2))
    frobenius_norm = jnp.sqrt(sum_sines_squared)
    return frobenius_norm

def reorthonormalize(C):
    """
    Reorthonormalize the coefficient matrix C using QR decomposition.
    """
    q, r = jnp.linalg.qr(C)
    return q

def objective_function(C_trial, C_CASSCF, S):
    """
    Objective function for the optimization.
    """
    sigma = overlap_matrix_svd_jax(C_CASSCF, C_trial, S)
    return frobenius_norm_sines_jax(sigma)

def clip_gradients(grads, clip_norm):
    """
    Clips the gradients to have a maximum norm of clip_norm.
    
    Parameters:
    - grads: Gradients to be clipped.
    - clip_norm: The maximum norm for the gradients.
    
    Returns:
    - Clipped gradients.
    """
    grad_norm = jnp.linalg.norm(grads)
    if grad_norm > clip_norm:
        grads = grads * (clip_norm / grad_norm)
    return grads

def project_onto_stiefel(C):
    """
    Project the matrix C onto the Stiefel manifold using SVD.
    
    Args:
    C: A matrix to be projected onto the Stiefel manifold.
    
    Returns:
    A matrix that is orthonormal and lies on the Stiefel manifold.
    """
    U, _, VT = svd(C, full_matrices=False)
    return U @ VT

def orthogonalize_with_overlap(C_trial, S):
    """
    Orthogonalizes C_trial considering the overlap matrix S.
    
    Args:
    - C_trial: The trial coefficient matrix.
    - S: The overlap matrix of atomic orbitals.
    
    Returns:
    - The adjusted coefficient matrix that satisfies the orthonormality constraint.
    """
    # Diagonalize the overlap matrix
    eigenvalues, eigenvectors = eigh(S)
    
    # Form the S^(-1/2) matrix
    S_inv_half = eigenvectors @ jnp.diag(1.0 / jnp.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Adjust C_trial to satisfy the orthonormality constraint
    C_adjusted = C_trial @ S_inv_half
    
    return C_adjusted

def modified_gram_schmidt(C, S):
    """
    Orthogonalize the columns of C with respect to the overlap matrix S using the Modified Gram-Schmidt process.
    
    Args:
    - C: The matrix of vectors to be orthogonalized (shape: (n, m)).
    - S: The overlap matrix (shape: (n, n)).
    
    Returns:
    - Q: The orthogonalized matrix of vectors (shape: (n, m)).
    """
    n, m = C.shape
    Q = jnp.zeros_like(C)
    
    for i in range(m):
        # Start with the original vector
        q = C[:, i]
        
        # Subtract projections onto previously orthogonalized vectors
        for j in range(i):
            # Compute the projection of q onto the j-th orthogonalized vector
            # Adjusting the inner product to include S: <q|S|Q[:, j]>
            S_proj = Q[:, j] @ S
            proj_coeff = (S_proj @ q) / (S_proj @ Q[:, j])
            q = q - proj_coeff * Q[:, j]
        
        # Normalize q with respect to S
        # Adjusting the norm to include S: sqrt(<q|S|q>)
        q_norm = jnp.sqrt(q.T @ S @ q)
        
        # Update the Q matrix using .at[].set() method
        Q = Q.at[:, i].set(q / q_norm)
    
    return Q

def modified_gram_schmidt_np(C, S):
    """
    Orthogonalize the columns of C with respect to the overlap matrix S using the Modified Gram-Schmidt process.
    
    Args:
    - C: The matrix of vectors to be orthogonalized (shape: (n, m)).
    - S: The overlap matrix (shape: (n, n)).
    
    Returns:
    - Q: The orthogonalized matrix of vectors (shape: (n, m)).
    """
    n, m = C.shape
    Q = np.zeros_like(C)
    
    for i in range(m):
        q = C[:, i]
        
        for j in range(i):
            S_proj = Q[:, j] @ S
            proj_coeff = (S_proj @ q) / (S_proj @ Q[:, j])
            q = q - proj_coeff * Q[:, j]
        
        q_norm = np.sqrt(q.T @ S @ q)
        Q[:, i] = q / q_norm
    
    return Q


# Define H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='3-21g', spin=0, charge=0)

# Perform RHF calculation
mf = scf.RHF(mol)
mf.kernel()

mci = mcscf.CASCI(mf, 2, 2)
mci.kernel()

# Perform CASSCF calculation with 2 orbitals and 2 electrons in active space
mc = mcscf.CASSCF(mf, 2, 2)
mc.kernel()
ncore = mc.ncore
print("The number of core orbitals is:", ncore)
#Obtain the overlap matrix S
S = mol.intor('int1e_ovlp')

# Extract CASSCF and trial (RHF) coefficient matrices for the active space
C_CASSCF = mc.mo_coeff[:, 0:mc.ncore + mc.ncas]
C_trial_initial = mf.mo_coeff[:, 0:mc.ncore + mc.ncas]

# Generate cube files for the orbitals before optimization
# iterate over orbitals and generate cube files
for i in range(C_trial_initial.shape[1]):
    # Generate the i-th molecular orbital
    mo = C_trial_initial[:, i]
    # Generate the cube file for the i-th molecular orbital
    cubegen.orbital(mol, f'initial_orbital_{i}.cube', mo, nx=100, ny=100, nz=100)


sigma_intial = overlap_matrix_svd_jax(C_CASSCF, C_trial_initial, S)
frobenius_norm_initial = frobenius_norm_sines_jax(sigma_intial)

print("Singular values", sigma_intial)
print("Frobnius norm of the matrix of sines of the principal angles:", frobenius_norm_initial)

# Set the optimization parameters
clip_norm = 50  # Example value; needs adjustment
convergence_threshold = 1e-2 # Example value; needs adjustment
gradient_function = grad(objective_function, argnums=0)  # Gradient w.r.t. the first argument
num_iterations = 10234
frobenius_norms = []
learning_rate = 1e-2

# Optimization sketch

for iteration in range(num_iterations):
    grads = gradient_function(C_trial_initial, C_CASSCF, S)

    # Clip the gradients
    grads = clip_gradients(grads, clip_norm)
    #print(grads)
    C_trial_initial -= learning_rate * grads    
    
    # Re-orthonormalize
    C_trial_initial = modified_gram_schmidt(C_trial_initial, S)
    
    # Compute the Frobenius norm for the current step
    sigma_current = overlap_matrix_svd_jax(C_CASSCF, C_trial_initial, S)
    frobenius_norm_current = frobenius_norm_sines_jax(sigma_current)

    # Save the frobenius norm for plotting
    frobenius_norms.append(frobenius_norm_current)
    # Print the gradient norm and the Frobenius norm at the current step
    print(f"Step {iteration + 1}, Gradient Norm: {jnp.linalg.norm(grads)}, Frobenius Norm: {frobenius_norm_current}, sigma: {sigma_current}")

    # Check for convergence
    if frobenius_norm_current < convergence_threshold:
        print(f"Converged at step {iteration + 1} with Frobenius norm: {frobenius_norm_current}")
        break

    # Generate cube files for the orbitals after optimization

# Save the optimized orbitals to a cube file

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'Trebuchet MS'
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams["axes.prop_cycle"]

# Plotting the Frobenius norms
plt.plot(frobenius_norms, marker='^',color='b')
plt.xlabel('Iteration')
plt.ylabel('Frobenius Norm')
plt.title('Frobenius Norm vs. Iteration')
#plt.grid(True)
plt.savefig('frobenius_norm_opt.png')

# iterate over orbitals and generate cube files
for i in range(C_trial_initial.shape[1]):
    # Generate the i-th molecular orbital
    mo = C_trial_initial[:, i]
    # Generate the cube file for the i-th molecular orbital
    cubegen.orbital(mol, f'optimized_orbital_{i}.cube', mo, nx=100, ny=100, nz=100)

# Generate cube files for the CASSCF orbitals
    
# iterate over orbitals and generate cube files
for i in range(C_CASSCF.shape[1]):
    # Generate the i-th molecular orbital
    mo = C_CASSCF[:, i]
    # Generate the cube file for the i-th molecular orbital
    cubegen.orbital(mol, f'CASSCF_orbital_{i}.cube', mo, nx=100, ny=100, nz=100)

# Contatinate the optimized C_trial_intial with the virtual orbitals with indices from ncore + ncas to nmo. Write this in numpy

C_trial_optimized = np.concatenate((C_trial_initial, mf.mo_coeff[:, ncore + mc.ncas:]), axis=1)

# orthonormalize the orbitals using the modified Gram-Schmidt process
C_trial_optimized = modified_gram_schmidt_np(C_trial_optimized, S)

# Convert C_trial_optimized to a numpy matrix



# Perform a CASCI calculation with the optimized orbitals
mc = mcscf.CASCI(mf, 2, 2)
mc.kernel(C_trial_optimized)

