from pyscf import gto, scf, mcscf
import numpy as np
from pyscf.tools import cubegen
import matplotlib.pyplot as plt


def overlap_matrix_svd(C_CASSCF, C_trial, S):
    """Compute the singular values of the overlap matrix."""
    M = np.dot(C_CASSCF.T.conj(), np.dot(S, C_trial))
    _, sigma, _ = np.linalg.svd(M)
    return sigma


def frobenius_norm_sines(sigma):
    """Compute the Frobenius norm of the matrix of sines of the principal angles."""
    sum_sines_squared = np.sum((1 - sigma**2))
    if sum_sines_squared < 0 and sum_sines_squared > -1e-5:
        sum_sines_squared = 0
    return np.sqrt(sum_sines_squared)


def reorthonormalize(C):
    """Reorthonormalize the coefficient matrix C using QR decomposition."""
    q, r = np.linalg.qr(C)
    return q


def clip_gradients(grads, clip_norm):
    """Clips the gradients to have a maximum norm of clip_norm."""
    grad_norm = np.linalg.norm(grads)
    if grad_norm > clip_norm:
        grads = grads * (clip_norm / grad_norm)
    return grads


def project_onto_stiefel(C):
    """Project the matrix C onto the Stiefel manifold using SVD."""
    U, _, VT = np.linalg.svd(C, full_matrices=False)
    return np.dot(U, VT)


def orthogonalize_with_overlap(C_trial, S):
    """Orthogonalizes C_trial considering the overlap matrix S."""
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    S_inv_half = np.dot(eigenvectors, np.dot(np.diag(1.0 / np.sqrt(eigenvalues)), eigenvectors.T))
    return np.dot(C_trial, S_inv_half)


def modified_gram_schmidt(C, S):
    """Orthogonalize the columns of C with respect to the overlap matrix S using the Modified Gram-Schmidt process."""
    n, m = C.shape
    Q = np.zeros_like(C)
    for i in range(m):
        q = C[:, i]
        for j in range(i):
            S_proj = np.dot(Q[:, j].T, S)
            proj_coeff = np.dot(S_proj, q) / np.dot(S_proj, Q[:, j])
            q = q - proj_coeff * Q[:, j]
        q_norm = np.sqrt(np.dot(q.T, np.dot(S, q)))
        Q[:, i] = q / q_norm
    return Q


def compute_energy_difference(C_trial_np, C_CASSCF_np):
    """Compute the energy difference between CASCI energy with trial coefficients and CASSCF energy with optimized coefficients using NumPy arrays."""
    mc_trial = mcscf.CASCI(mf, mc.ncas, mc.nelecas)
    e_casci = mc_trial.kernel(C_trial_np)[0]
    e_casscf = mc.kernel(C_CASSCF_np)[0]
    return e_casci - e_casscf


def finite_difference_gradient(C_trial, C_CASSCF, S, epsilon=1e-5):
    """Compute the gradient of the objective function using finite differences."""
    grad = np.zeros_like(C_trial)
    for i in range(C_trial.shape[0]):
        for j in range(C_trial.shape[1]):
            C_trial_pos = C_trial.copy()
            C_trial_neg = C_trial.copy()
            C_trial_pos[i, j] += epsilon
            C_trial_neg[i, j] -= epsilon
            obj_pos = objective_function(C_trial_pos, C_CASSCF, S, lambda_reg)
            obj_neg = objective_function(C_trial_neg, C_CASSCF, S, lambda_reg)
            grad[i, j] = (obj_pos - obj_neg) / (2 * epsilon)
    return grad


def objective_function(C_trial, C_CASSCF, S, lambda_reg):
    """Objective function for the optimization with regularization."""
    sigma = overlap_matrix_svd(C_CASSCF, C_trial, S)
    frobenius_norm = frobenius_norm_sines(sigma)
    energy_diff = compute_energy_difference(C_trial, C_CASSCF)
    return frobenius_norm + np.abs(energy_diff) + lambda_reg * np.linalg.norm(C_trial - C_CASSCF) ** 2


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

# Obtain the overlap matrix S
S = mol.intor('int1e_ovlp')

# Extract CASSCF and trial (RHF) coefficient matrices for the active space
C_CASSCF = mc.mo_coeff[:, 0:mc.ncore + mc.ncas]
C_trial_initial = mf.mo_coeff[:, 0:mc.ncore + mc.ncas]

# Generate cube files for the orbitals before optimization
for i in range(C_trial_initial.shape[1]):
    mo = C_trial_initial[:, i]
    cubegen.orbital(mol, f'initial_orbital_{i}.cube', mo, nx=100, ny=100, nz=100)

sigma_initial = overlap_matrix_svd(C_CASSCF, C_trial_initial, S)
frobenius_norm_initial = frobenius_norm_sines(sigma_initial)

print("Singular values", sigma_initial)
print("Frobenius norm of the matrix of sines of the principal angles:", frobenius_norm_initial)

import numpy as np

# Optimization loop parameters
num_iterations = 100000
initial_learning_rate = 1e-2
frobenius_norms = []
convergence_threshold = 1e-2
clip_norm = 50
epsilon_adagrad = 1e-8  # Smoothing term to prevent division by zero

# Initialize the squared gradient accumulation for Adagrad
grad_squared_accum = np.zeros_like(C_trial_initial)

# Regularization parameter
lambda_reg = 1e-3

for iteration in range(num_iterations):
    # Compute the objective function and its gradient using finite differences
    obj = objective_function(C_trial_initial, C_CASSCF, S, lambda_reg)
    grads = finite_difference_gradient(C_trial_initial, C_CASSCF, S)

    # Clip the gradients (optional)
    grads = clip_gradients(grads, clip_norm)

    # Accumulate squared gradients for Adagrad
    grad_squared_accum += grads ** 2

    # Compute adaptive learning rate
    learning_rate = initial_learning_rate / (np.sqrt(grad_squared_accum) + epsilon_adagrad)

    # Update the trial coefficients using adaptive learning rate
    C_trial_initial -= learning_rate * grads

    # Re-orthonormalize
    C_trial_initial = modified_gram_schmidt(C_trial_initial, S)

    # Compute the Frobenius norm for the current step
    sigma_current = overlap_matrix_svd(C_CASSCF, C_trial_initial, S)
    frobenius_norm_current = frobenius_norm_sines(sigma_current)

    # Save the Frobenius norm for plotting
    frobenius_norms.append(frobenius_norm_current)

    # Print the gradient norm and the Frobenius norm at the current step
    print(f"Step {iteration + 1}, Gradient Norm: {np.linalg.norm(grads)}, Frobenius Norm: {frobenius_norm_current}, sigma: {sigma_current}")

    # Check for convergence
    if frobenius_norm_current < convergence_threshold:
        print(f"Converged at step {iteration + 1} with Frobenius norm: {frobenius_norm_current}")
        break

# Plotting the Frobenius norms with markersize small
plt.plot(frobenius_norms, marker='^', color='r', )
plt.xlabel('Iteration')
plt.ylabel('Frobenius Norm')
plt.title('Frobenius Norm vs. Iteration')
plt.savefig('frobenius_norm_opt_v7.png')

# Generate cube files for the orbitals after optimization
for i in range(C_trial_initial.shape[1]):
    mo = C_trial_initial[:, i]
    cubegen.orbital(mol, f'optimized_orbital_{i}.cube', mo, nx=100, ny=100, nz=100)

# Generate cube files for the CASSCF orbitals
for i in range(C_CASSCF.shape[1]):
    mo = C_CASSCF[:, i]
    cubegen.orbital(mol, f'CASSCF_orbital_{i}.cube', mo, nx=100, ny=100, nz=100)

# Concatenate the optimized C_trial_initial with the virtual orbitals
C_trial_optimized = np.concatenate((C_trial_initial, mf.mo_coeff[:, ncore + mc.ncas:]), axis=1)

# Orthonormalize the orbitals using the modified Gram-Schmidt process
C_trial_optimized = modified_gram_schmidt(C_trial_optimized, S)

# Perform a CASCI calculation with the optimized orbitals
mc = mcscf.CASCI(mf, 2, 2)
mc.kernel(C_trial_optimized)
