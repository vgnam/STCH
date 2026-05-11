"""
Smooth-Tchebycheff Joint MCMC for Multi-Objective Optimization

Implementation of the Bayesian MOO framework described in AGENT.md:
- Posterior distribution over the Pareto front via joint MCMC over (x, λ)
- Smooth Tchebycheff as the energy function
- Langevin dynamics for x-updates
- Projected Langevin on the simplex for λ-updates
- Annealing schedules for μ (smoothing) and T (temperature)
"""

import os
import numpy as np
import torch
import timeit
import matplotlib.pyplot as plt

from problem import get_problem
from pymoo.indicators.hv import HV

# ------------------------------------------------------------------------------
# Tunable hyper-parameters
# ------------------------------------------------------------------------------

# Benchmark problems to run
INS_LIST = ['f1']  # quick test

# MCMC parameters
N_PARTICLES = 20          # number of particles
N_ITER = 200              # total MCMC iterations
BURN_IN_FRACTION = 0.5    # collect samples after this fraction of iterations

# Step sizes (ε_x < ε_λ as per AGENT.md)
EPS_X = 1e-3
EPS_LAMBDA = 5e-3

# Annealing schedules
MU_START = 0.01
MU_END = 30.0
T_START = 5.0
T_END = 0.005

# Dirichlet prior concentration for λ
ALPHA = 2.0

# Augmented Tchebycheff parameter (0.0 = disabled)
RHO = 0.0

# Gradient clipping for stability
GRAD_CLIP = 100.0

# Device
device = 'cpu'

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def project_simplex(v):
    """Project a vector onto the probability simplex.

    Algorithm (from AGENT.md):
        1. Sort v in descending order -> u
        2. Find largest index rho such that:
             u_rho > (1/rho) * (sum_{j=1}^{rho} u_j - 1)
        3. theta = (1/rho) * (sum_{j=1}^{rho} u_j - 1)
        4. lambda_k = max(v_k - theta, 0)

    Args:
        v: 1-D array of length K.

    Returns:
        w: projected vector on the probability simplex.
    """
    v = np.asarray(v, dtype=np.float64)
    K = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(K) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones(K) / K
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w


def mu_schedule(t, n_iter, mu_start=MU_START, mu_end=MU_END):
    """Annealing schedule for smoothing parameter mu.

    Phase 1 (exploration, first half): mu small -> medium
    Phase 2 (exploitation, second half): mu medium -> large
    """
    half = n_iter // 2
    if half == 0:
        return mu_end
    if t <= half:
        progress = t / half
        return mu_start + (mu_end * 0.4 - mu_start) * progress
    else:
        progress = (t - half) / half
        return mu_end * 0.4 + (mu_end - mu_end * 0.4) * progress


def T_schedule(t, n_iter, T_start=T_START, T_end=T_END):
    """Annealing schedule for temperature T.

    Phase 1 (exploration): T large -> medium
    Phase 2 (exploitation): T medium -> small
    """
    half = n_iter // 2
    if half == 0:
        return T_end
    if t <= half:
        progress = t / half
        return T_start + (T_end * 20 - T_start) * progress
    else:
        progress = (t - half) / half
        return T_end * 20 + (T_end - T_end * 20) * progress


# ------------------------------------------------------------------------------
# Main sampler class
# ------------------------------------------------------------------------------


class SmoothTchMCMC:
    """Smooth-Tchebycheff Joint MCMC sampler."""

    def __init__(self, problem, N, n_iter, alpha=ALPHA,
                 eps_x=EPS_X, eps_lambda=EPS_LAMBDA,
                 mu_start=MU_START, mu_end=MU_END,
                 T_start=T_START, T_end=T_END,
                 device='cpu', rho=RHO, normalize=True,
                 ideal_point=None, nadir_point=None,
                 burn_in_fraction=BURN_IN_FRACTION):
        self.problem = problem
        self.N = N
        self.n_iter = n_iter
        self.burn_in_start = int(n_iter * burn_in_fraction) + 1
        self.alpha = alpha
        self.eps_x = eps_x
        self.eps_lambda = eps_lambda
        self.mu_start = mu_start
        self.mu_end = mu_end
        self.T_start = T_start
        self.T_end = T_end
        self.device = device
        self.rho = rho
        self.normalize = normalize

        self.n_dim = problem.n_dim
        self.n_obj = problem.n_obj

        # Normalization points
        if normalize:
            if ideal_point is None:
                self.ideal_point = torch.zeros(self.n_obj, device=device)
            else:
                self.ideal_point = torch.tensor(
                    ideal_point, device=device, dtype=torch.float64)
            if nadir_point is None:
                self.nadir_point = torch.ones(self.n_obj, device=device)
            else:
                self.nadir_point = torch.tensor(
                    nadir_point, device=device, dtype=torch.float64)
        else:
            self.ideal_point = None
            self.nadir_point = None

        # Initialize particles in [0, 1] (same convention as ParetoSetModel)
        self.X = torch.rand(N, self.n_dim, device=device, dtype=torch.float64)

        # Initialize preferences uniformly on simplex
        lam = np.zeros((N, self.n_obj))
        for i in range(N):
            lam[i] = np.random.dirichlet(np.ones(self.n_obj))
        self.Lambda = torch.tensor(lam, device=device, dtype=torch.float64)

        # Utopia point z* (best known value per objective)
        self.z_star = torch.full(
            (self.n_obj,), float('inf'), device=device, dtype=torch.float64)

        # Storage for post-burn-in samples
        self.samples_X = []
        self.samples_Lambda = []
        self.samples_F = []

    def normalize_values(self, f_vals):
        """Normalize objective values using ideal / nadir points."""
        if self.normalize and self.nadir_point is not None:
            return (f_vals - self.ideal_point) / (self.nadir_point - self.ideal_point)
        return f_vals

    def evaluate_objectives(self, X):
        """Evaluate and normalize objectives.

        Args:
            X: (M, n_dim) tensor in [0, 1].

        Returns:
            f_vals: (M, n_obj) normalized objective values.
        """
        f_vals = self.problem.evaluate(X)  # (M, n_obj)
        return self.normalize_values(f_vals)

    def step(self, t):
        """Perform one MCMC iteration."""
        mu = mu_schedule(t, self.n_iter, self.mu_start, self.mu_end)
        T = T_schedule(t, self.n_iter, self.T_start, self.T_end)

        # ----------------------------------------------------------
        # 1. Evaluate objectives (no grad) and update utopia point
        # ----------------------------------------------------------
        with torch.no_grad():
            f_vals = self.evaluate_objectives(self.X)  # (N, K)

        # Guard: re-initialize any particle that produced NaN / Inf
        valid_mask = torch.isfinite(f_vals).all(dim=1)
        if not valid_mask.all():
            bad_idx = torch.where(~valid_mask)[0]
            self.X[bad_idx] = torch.rand(
                len(bad_idx), self.n_dim,
                device=self.device, dtype=torch.float64
            )
            with torch.no_grad():
                f_vals = self.evaluate_objectives(self.X)

        min_vals = f_vals.min(dim=0)[0]
        if torch.isfinite(min_vals).all():
            self.z_star = torch.minimum(self.z_star, min_vals)

        # ----------------------------------------------------------
        # 2. Update each particle
        # ----------------------------------------------------------
        for i in range(self.N):
            f_i = f_vals[i]           # (K,) — cached for λ-update
            lam_i = self.Lambda[i]    # (K,)
            z = self.z_star           # (K,)

            # --- x-update via Langevin dynamics -------------------
            # Slight boundary padding prevents √0-style NaN gradients
            x_i = torch.clamp(
                self.X[i:i + 1].clone(), 1e-6, 1.0 - 1e-6
            ).requires_grad_(True)
            f_i_grad = self.evaluate_objectives(x_i)[0]  # (K,)

            # Adaptive weights: w_k = softmax_k(mu * lambda_k * (f_k - z_k))
            a = mu * lam_i * (f_i_grad - z)
            w = torch.softmax(a, dim=0).detach()

            # Scalar whose gradient gives the score direction
            # grad_x L_smooth = sum_k w_k * lambda_k * grad_x f_k
            # grad_x L_aug    = rho * sum_k lambda_k * grad_x f_k
            scalar = torch.sum((w + self.rho) * lam_i * f_i_grad)

            score_x = -(1.0 / T) * torch.autograd.grad(
                scalar, x_i, create_graph=False
            )[0][0]

            # Gradient clipping & NaN guard
            score_x = torch.where(
                torch.isfinite(score_x),
                torch.clamp(score_x, -GRAD_CLIP, GRAD_CLIP),
                torch.zeros_like(score_x)
            )

            # Langevin update
            noise_x = torch.randn(
                self.n_dim, device=self.device, dtype=torch.float64)
            self.X[i] = (
                self.X[i]
                + (self.eps_x / 2.0) * score_x
                + np.sqrt(self.eps_x) * noise_x
            )
            # Clip to feasible box [0, 1] (same convention as PSL)
            self.X[i] = torch.clamp(self.X[i], 0.0, 1.0)

            # --- λ-update via Projected Langevin on simplex -------
            # score_lambda = -(1/T) * (w + rho) * (f - z) + (alpha-1)/lambda
            score_lambda = -(1.0 / T) * (w + self.rho) * (f_i - z)
            if self.alpha != 1.0:
                score_lambda = score_lambda + (self.alpha - 1.0) / (lam_i + 1e-20)

            noise_lambda = torch.randn(
                self.n_obj, device=self.device, dtype=torch.float64)
            lam_proposal = (
                lam_i
                + (self.eps_lambda / 2.0) * score_lambda
                + np.sqrt(self.eps_lambda) * noise_lambda
            )

            # Project back to simplex
            self.Lambda[i] = torch.tensor(
                project_simplex(lam_proposal.cpu().numpy()),
                device=self.device, dtype=torch.float64
            )

        # ----------------------------------------------------------
        # 3. Collect samples after burn-in
        # ----------------------------------------------------------
        if t >= self.burn_in_start:
            self.samples_X.append(self.X.cpu().clone().numpy())
            self.samples_Lambda.append(self.Lambda.cpu().clone().numpy())
            self.samples_F.append(f_vals.cpu().clone().numpy())

        return mu, T

    def run(self):
        """Run the full MCMC chain."""
        for t in range(1, self.n_iter + 1):
            mu, T = self.step(t)
            if t % 200 == 0 or t == 1:
                z_str = np.array2string(
                    self.z_star.cpu().numpy(), precision=4, suppress_small=True)
                print(f"  Iter {t:5d}/{self.n_iter} | mu={mu:.4f} | "
                      f"T={T:.6f} | z*={z_str}")
        print("  Sampling complete.")

    def get_collected_samples(self):
        """Return all post-burn-in samples as flattened arrays."""
        if len(self.samples_X) == 0:
            return None, None, None
        X_all = np.concatenate(self.samples_X, axis=0)
        Lambda_all = np.concatenate(self.samples_Lambda, axis=0)
        F_all = np.concatenate(self.samples_F, axis=0)
        return X_all, Lambda_all, F_all

    def compute_metrics(self, ref_point=None):
        """Compute evaluation metrics on collected samples.

        Returns:
            metrics: dict with hypervolume, diversity, uncertainty.
            pf: non-dominated objective vectors (M, n_obj).
        """
        X_all, Lambda_all, F_all = self.get_collected_samples()
        if F_all is None:
            return {}, None

        # Subsample if too many points for expensive dominance check
        M = F_all.shape[0]
        if M > 5000:
            idx = np.random.choice(M, 5000, replace=False)
            F_sub = F_all[idx]
        else:
            F_sub = F_all

        # Extract non-dominated front (naive O(M^2) — fine for M <= 5000)
        is_nd = np.ones(F_sub.shape[0], dtype=bool)
        for i in range(F_sub.shape[0]):
            if not is_nd[i]:
                continue
            better = np.all(F_sub <= F_sub[i], axis=1) & np.any(F_sub < F_sub[i], axis=1)
            if np.any(better):
                is_nd[i] = False
        pf = F_sub[is_nd]

        metrics = {}

        # Hypervolume
        if ref_point is None:
            if self.normalize:
                ref_point = np.array([1.1] * self.n_obj)
            else:
                ref_point = np.max(F_all, axis=0) + 0.1
        if len(pf) > 0:
            hv = HV(ref_point=ref_point)
            metrics['hypervolume'] = float(hv(pf))

        # Diversity: average pairwise Euclidean distance on Pareto front
        if len(pf) > 1:
            # Sample at most 500 points for diversity to keep it fast
            if len(pf) > 500:
                idx = np.random.choice(len(pf), 500, replace=False)
                pf_div = pf[idx]
            else:
                pf_div = pf
            dists = []
            for i in range(len(pf_div)):
                for j in range(i + 1, len(pf_div)):
                    dists.append(np.linalg.norm(pf_div[i] - pf_div[j]))
            metrics['diversity'] = float(np.mean(dists))
        else:
            metrics['diversity'] = 0.0

        # Uncertainty per objective (std across all collected samples)
        metrics['uncertainty'] = np.std(F_all, axis=0).tolist()

        return metrics, pf


# ------------------------------------------------------------------------------
# Plotting helpers (style matches run_stch_psl.py)
# ------------------------------------------------------------------------------


def setup_plot_style():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 15
    MAX_SIZE = 18
    plt.rc('font', family='Times New Roman', size=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MAX_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_2d_pf(pf, test_ins, save_path=None):
    fig = plt.figure()
    plt.scatter(pf[:, 0], pf[:, 1], c='tomato', alpha=0.6, s=30, label='STCH-MCMC')
    plt.xlabel(r'$f_1(x)$', size=16)
    plt.ylabel(r'$f_2(x)$', size=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.title(f'{test_ins} — Pareto Front')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_3d_pf(pf, test_ins, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2],
               c='tomato', s=30, alpha=0.6, label='STCH-MCMC')
    ax.set_xlabel(r'$f_1(x)$', size=12)
    ax.set_ylabel(r'$f_2(x)$', size=12)
    ax.set_zlabel(r'$f_3(x)$', size=12)
    ax.legend(loc=1, bbox_to_anchor=(1, 1))
    plt.title(f'{test_ins} — Pareto Front')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------------------
# Main execution loop
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    setup_plot_style()

    for test_ins in INS_LIST:
        print("\n" + "=" * 70)
        print(f"Problem: {test_ins}")
        print("=" * 70)

        # Load problem
        problem = get_problem(test_ins)
        n_obj = problem.n_obj

        # Load normalization points for RE problems
        if test_ins.startswith('re'):
            base = os.path.dirname(os.path.abspath(__file__))
            ideal = np.loadtxt(
                os.path.join(base, f'data/RE/ideal_nadir_points/ideal_point_{test_ins}.dat'))
            nadir = np.loadtxt(
                os.path.join(base, f'data/RE/ideal_nadir_points/nadir_point_{test_ins}.dat'))
        else:
            ideal = np.zeros(n_obj)
            nadir = np.ones(n_obj)

        # Build and run sampler
        sampler = SmoothTchMCMC(
            problem=problem,
            N=N_PARTICLES,
            n_iter=N_ITER,
            alpha=ALPHA,
            eps_x=EPS_X,
            eps_lambda=EPS_LAMBDA,
            mu_start=MU_START,
            mu_end=MU_END,
            T_start=T_START,
            T_end=T_END,
            device=device,
            rho=RHO,
            normalize=True,
            ideal_point=ideal,
            nadir_point=nadir,
        )

        start = timeit.default_timer()
        sampler.run()
        stop = timeit.default_timer()

        # Metrics
        metrics, pf = sampler.compute_metrics()
        print(f"\n  Time: {stop - start:.2f}s")
        for k, v in metrics.items():
            if k == 'uncertainty':
                print(f"  {k:15s}: {np.array(v)}")
            else:
                print(f"  {k:15s}: {v:.6f}")

        # Plotting
        if pf is not None and len(pf) > 0:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{test_ins}_mcmc.png')

            if n_obj == 2:
                plot_2d_pf(pf, test_ins, save_path=save_path)
            elif n_obj == 3:
                plot_3d_pf(pf, test_ins, save_path=save_path)

            print(f"  Plot saved to: {save_path}")

        print("\n" + "*" * 70)
