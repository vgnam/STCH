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
from model import ParetoSetModel
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

# ------------------------------------------------------------------------------
# Tunable hyper-parameters
# ------------------------------------------------------------------------------

# Benchmark problems to run
INS_LIST = ['f1']  # quick test

# MCMC parameters
N_PARTICLES = 64          # number of particles
N_ITER = 600              # total MCMC iterations
BURN_IN_FRACTION = 0.5    # collect samples after this fraction of iterations

# PSL-STCH baseline parameters for comparison
PSL_N_STEPS = 2000
PSL_N_PREF_UPDATE = 10
PSL_LR = 1e-3
PSL_MU = 0.01

# Step sizes (ε_x < ε_λ as per AGENT.md)
EPS_X = 5e-3
EPS_LAMBDA = 2e-3

# Zero-temperature refinement approximates the posterior mode as T -> 0.
REFINE_STEPS = 200
REFINE_LR = 5e-3

# Annealing schedules for the repo's STCH convention:
# L_mu = mu * logsumexp(lambda * (f - z*) / mu), so smaller mu is sharper.
MU_START = 0.20
MU_END = 0.01
T_START = 0.20
T_END = 0.01

# Dirichlet prior concentration for λ
ALPHA = 1.0
LAMBDA_EPS = 1e-3

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


def make_simplex_interior(v, eps=LAMBDA_EPS):
    """Move simplex vectors into the eps-interior to avoid corner degeneracy."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = np.maximum(v, eps)
        return v / np.sum(v)

    v = np.maximum(v, eps)
    return v / np.sum(v, axis=1, keepdims=True)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(
                ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1
            )


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)

    ref_dirs = []
    ref_dir = np.full(n_dim, np.nan)
    das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
    return np.concatenate(ref_dirs, axis=0)


def get_eval_preferences(n_obj):
    if n_obj == 2:
        grid = np.linspace(0, 1, 200)
        return np.stack([grid, 1 - grid], axis=1)
    if n_obj == 3:
        return das_dennis(44, 3)
    raise ValueError(f"No preference grid configured for {n_obj} objectives.")


def get_stratified_preferences(n_points, n_obj):
    """Deterministic simplex coverage for preference particles."""
    if n_obj == 2:
        grid = np.linspace(0, 1, n_points)
        return make_simplex_interior(np.stack([grid, 1 - grid], axis=1))

    if n_obj == 3:
        n_partitions = 1
        while (n_partitions + 1) * (n_partitions + 2) // 2 < n_points:
            n_partitions += 1
        refs = das_dennis(n_partitions, n_obj)
        if len(refs) == n_points:
            return make_simplex_interior(refs)
        idx = np.linspace(0, len(refs) - 1, n_points).round().astype(int)
        return make_simplex_interior(refs[idx])

    return make_simplex_interior(np.random.dirichlet(np.ones(n_obj), n_points))


def get_reference_front(test_ins, n_obj, ideal_point, nadir_point, n_points=1000):
    """Return a normalized reference Pareto front for IGD."""
    if test_ins in ['f1', 'f2', 'f3']:
        x = np.linspace(0, 1, n_points)
        return np.stack([x, 1 - np.sqrt(x)], axis=1)

    if test_ins in ['f4', 'f5', 'f6']:
        x = np.linspace(0, 1, n_points)
        return np.stack([x, 1 - x ** 2], axis=1)

    if test_ins.startswith('re'):
        base = os.path.dirname(os.path.abspath(__file__))
        pf_path = os.path.join(base, f'data/RE/ParetoFront/{test_ins}.dat')
        if os.path.exists(pf_path):
            pf = np.loadtxt(pf_path)
            pf = np.atleast_2d(pf)
            return (pf - ideal_point) / (nadir_point - ideal_point)

    return None


def nondominated_mask(F):
    """Return a boolean mask for non-dominated minimization objective vectors."""
    if F is None or len(F) == 0:
        return np.array([], dtype=bool)

    is_nd = np.ones(F.shape[0], dtype=bool)
    for i in range(F.shape[0]):
        if not is_nd[i]:
            continue
        better = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(better):
            is_nd[i] = False
    return is_nd


def filter_nondominated(F):
    """Extract a non-dominated front from minimization objective vectors."""
    if F is None or len(F) == 0:
        return np.empty((0, 0))
    return F[nondominated_mask(F)]


def diversity(front):
    """Average pairwise Euclidean distance on a front."""
    if front is None or len(front) <= 1:
        return 0.0

    if len(front) > 500:
        idx = np.random.choice(len(front), 500, replace=False)
        front = front[idx]

    dists = []
    for i in range(len(front)):
        for j in range(i + 1, len(front)):
            dists.append(np.linalg.norm(front[i] - front[j]))
    return float(np.mean(dists)) if dists else 0.0


def compute_front_metrics(F, n_obj, ref_point=None, reference_pf=None):
    """Compute HV, IGD, and diversity from normalized objective vectors."""
    if F is None or len(F) == 0:
        return {'hypervolume': np.nan, 'igd': np.nan, 'diversity': 0.0}, F

    if ref_point is None:
        ref_point = np.array([1.1] * n_obj)

    pf = filter_nondominated(F)
    metrics = {
        'hypervolume': np.nan,
        'igd': np.nan,
        'diversity': diversity(pf),
    }

    if len(pf) > 0:
        metrics['hypervolume'] = float(HV(ref_point=ref_point)(pf))
        if reference_pf is not None and len(reference_pf) > 0:
            metrics['igd'] = float(IGD(reference_pf)(pf))

    return metrics, pf


def mu_schedule(t, n_iter, mu_start=MU_START, mu_end=MU_END):
    """Anneal from smooth weighted-sum-like STCH to sharper Tchebycheff."""
    if n_iter <= 1:
        return mu_end
    progress = (t - 1) / (n_iter - 1)
    return mu_start * (mu_end / mu_start) ** progress


def T_schedule(t, n_iter, T_start=T_START, T_end=T_END):
    """Geometric temperature annealing for Langevin exploration to exploitation."""
    if n_iter <= 1:
        return T_end
    progress = (t - 1) / (n_iter - 1)
    return T_start * (T_end / T_start) ** progress


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
        self.fixed_utopia = normalize and ideal_point is not None

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

        # Stratified simplex initialization improves preference-space coverage.
        lam = get_stratified_preferences(N, self.n_obj)
        self.Lambda = torch.tensor(lam, device=device, dtype=torch.float64)

        # Utopia point z*. With known ideal/nadir normalization, the normalized
        # utopia is exactly zero; otherwise estimate it online from particles.
        if self.fixed_utopia:
            self.z_star = torch.zeros(self.n_obj, device=device, dtype=torch.float64)
        else:
            self.z_star = torch.full(
                (self.n_obj,), float('inf'), device=device, dtype=torch.float64)

        # Storage for post-burn-in samples
        self.samples_X = []
        self.samples_Lambda = []
        self.samples_F = []

        # Optimizer-style output: best non-dominated points visited by the chain.
        self.archive_X = None
        self.archive_F = None

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

    def update_archive(self, X, F):
        """Keep the non-dominated archive of all visited particles."""
        X_np = X.detach().cpu().numpy()
        F_np = F.detach().cpu().numpy()
        finite_mask = np.isfinite(F_np).all(axis=1)
        if not np.any(finite_mask):
            return

        X_np = X_np[finite_mask]
        F_np = F_np[finite_mask]

        if self.archive_F is None:
            merged_X = X_np
            merged_F = F_np
        else:
            merged_X = np.concatenate([self.archive_X, X_np], axis=0)
            merged_F = np.concatenate([self.archive_F, F_np], axis=0)

        keep = nondominated_mask(merged_F)
        self.archive_X = merged_X[keep]
        self.archive_F = merged_F[keep]

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

        if not self.fixed_utopia:
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

            # Adaptive weights for L_mu = mu * logsumexp(lambda * (f - z) / mu).
            a = lam_i * (f_i_grad - z) / mu
            w = torch.softmax(a, dim=0).detach()

            # Scalar whose gradient gives the score direction
            # grad_x L_smooth = sum_k w_k * lambda_k * grad_x f_k
            # grad_x L_aug    = rho * sum_k lambda_k * grad_x f_k
            scalar = torch.sum((w + self.rho) * lam_i * f_i_grad)

            grad_x_energy = torch.autograd.grad(
                scalar, x_i, create_graph=False
            )[0][0]

            # Gradient clipping & NaN guard
            grad_x_energy = torch.where(
                torch.isfinite(grad_x_energy),
                torch.clamp(grad_x_energy, -GRAD_CLIP, GRAD_CLIP),
                torch.zeros_like(grad_x_energy)
            )

            # Temperature-consistent Langevin update for pi_T(x) proportional
            # to exp(-U(x) / T): drift follows -grad U and noise scales as sqrt(T).
            noise_x = torch.randn(
                self.n_dim, device=self.device, dtype=torch.float64)
            self.X[i] = (
                self.X[i]
                - self.eps_x * grad_x_energy
                + np.sqrt(2.0 * self.eps_x * T) * noise_x
            )
            # Clip to feasible box [0, 1] (same convention as PSL)
            self.X[i] = torch.clamp(self.X[i], 0.0, 1.0)

            # --- λ-update via Projected Langevin on simplex -------
            # Same temperature-consistent update on the preference simplex.
            delta_i = f_i_grad.detach() - z
            grad_lambda_energy = (w + self.rho) * delta_i
            prior_drift = torch.zeros_like(lam_i)
            if self.alpha != 1.0:
                prior_drift = (self.alpha - 1.0) / (lam_i + 1e-20)

            noise_lambda = torch.randn(
                self.n_obj, device=self.device, dtype=torch.float64)
            lam_proposal = (
                lam_i
                - self.eps_lambda * grad_lambda_energy
                + self.eps_lambda * T * prior_drift
                + np.sqrt(2.0 * self.eps_lambda * T) * noise_lambda
            )

            # Project back to simplex
            self.Lambda[i] = torch.tensor(
                make_simplex_interior(project_simplex(lam_proposal.cpu().numpy())),
                device=self.device, dtype=torch.float64
            )

        with torch.no_grad():
            current_f_vals = self.evaluate_objectives(self.X)
        self.update_archive(self.X, current_f_vals)

        # ----------------------------------------------------------
        # 3. Collect posterior samples after burn-in
        # ----------------------------------------------------------
        if t >= self.burn_in_start:
            self.samples_X.append(self.X.cpu().clone().numpy())
            self.samples_Lambda.append(self.Lambda.cpu().clone().numpy())
            self.samples_F.append(current_f_vals.cpu().clone().numpy())

        return mu, T

    def refine_particles(self, n_steps=REFINE_STEPS, lr=REFINE_LR):
        """Zero-temperature STCH refinement of final particles.

        This is the deterministic limit of annealed Langevin dynamics and is
        used only to improve the optimizer archive, not posterior uncertainty.
        """
        if n_steps <= 0:
            return

        x_param = self.X.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_param], lr=lr)
        mu = self.mu_end
        z = self.z_star.detach()
        lam = self.Lambda.detach()

        for _ in range(n_steps):
            optimizer.zero_grad()
            x_clamped = torch.clamp(x_param, 1e-6, 1.0 - 1e-6)
            f_vals = self.evaluate_objectives(x_clamped)
            delta = f_vals - z
            smooth = mu * torch.logsumexp(lam * delta / mu, dim=1)
            augmented = self.rho * torch.sum(lam * delta, dim=1)
            loss = torch.sum(smooth + augmented)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_param.clamp_(0.0, 1.0)

        self.X = x_param.detach()
        with torch.no_grad():
            refined_f_vals = self.evaluate_objectives(self.X)
        self.update_archive(self.X, refined_f_vals)

    def run(self):
        """Run the full MCMC chain."""
        for t in range(1, self.n_iter + 1):
            mu, T = self.step(t)
            if t % 200 == 0 or t == 1:
                z_str = np.array2string(
                    self.z_star.cpu().numpy(), precision=4, suppress_small=True)
                print(f"  Iter {t:5d}/{self.n_iter} | mu={mu:.4f} | "
                      f"T={T:.6f} | z*={z_str}")
        self.refine_particles()
        print("  Sampling complete.")

    def get_collected_samples(self):
        """Return all post-burn-in samples as flattened arrays."""
        if len(self.samples_X) == 0:
            return None, None, None
        X_all = np.concatenate(self.samples_X, axis=0)
        Lambda_all = np.concatenate(self.samples_Lambda, axis=0)
        F_all = np.concatenate(self.samples_F, axis=0)
        return X_all, Lambda_all, F_all

    def compute_metrics(self, ref_point=None, reference_pf=None, use_archive=True):
        """Compute evaluation metrics on collected samples.

        Returns:
            metrics: dict with hypervolume, IGD, diversity, uncertainty.
            pf: non-dominated objective vectors (M, n_obj).
        """
        X_all, Lambda_all, F_all = self.get_collected_samples()
        if F_all is None:
            return {}, None

        metric_F = self.archive_F if use_archive and self.archive_F is not None else F_all

        # Subsample if too many points for expensive dominance check.
        M = metric_F.shape[0]
        if M > 5000:
            idx = np.random.choice(M, 5000, replace=False)
            F_sub = metric_F[idx]
        else:
            F_sub = metric_F

        if ref_point is None:
            if self.normalize:
                ref_point = np.array([1.1] * self.n_obj)
            else:
                ref_point = np.max(F_all, axis=0) + 0.1

        metrics, pf = compute_front_metrics(
            F_sub, self.n_obj, ref_point=ref_point, reference_pf=reference_pf
        )

        # Uncertainty per objective (std across all collected samples)
        metrics['uncertainty'] = np.std(F_all, axis=0).tolist()
        metrics['archive_size'] = 0 if self.archive_F is None else len(self.archive_F)

        return metrics, pf


# ------------------------------------------------------------------------------
# PSL-STCH baseline
# ------------------------------------------------------------------------------


def run_psl_stch(problem, n_steps, n_pref_update, ideal_point, nadir_point, device='cpu'):
    """Run the original Pareto Set Learning STCH baseline."""
    psmodel = ParetoSetModel(problem.n_dim, problem.n_obj).to(device)
    optimizer = torch.optim.Adam(psmodel.parameters(), lr=PSL_LR)

    ideal = torch.tensor(ideal_point, device=device, dtype=torch.float64)
    nadir = torch.tensor(nadir_point, device=device, dtype=torch.float64)
    z = torch.zeros(problem.n_obj, device=device, dtype=torch.float64)

    for _ in range(n_steps):
        psmodel.train()

        pref = np.random.dirichlet(np.ones(problem.n_obj), n_pref_update)
        pref_vec = torch.tensor(pref, device=device, dtype=torch.float32)

        x = psmodel(pref_vec)
        value = problem.evaluate(x)
        value = (value - ideal) / (nadir - ideal)

        stch_value = PSL_MU * torch.logsumexp(pref_vec * (value - z) / PSL_MU, dim=1)
        loss = torch.sum(stch_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        psmodel.eval()
        pref = torch.tensor(
            get_eval_preferences(problem.n_obj), device=device, dtype=torch.float32
        )
        sol = psmodel(pref)
        obj = problem.evaluate(sol)
        obj_norm = (obj - ideal) / (nadir - ideal)

    return sol.cpu().numpy(), obj_norm.cpu().numpy()


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


def _metric_text(value):
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f'{value:.4f}'


def save_figure(fig, save_path):
    """Save a figure, avoiding crashes when an old plot file is locked."""
    try:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        return save_path
    except PermissionError:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_dir = os.path.join(repo_root, 'comparison_plots')
        os.makedirs(fallback_dir, exist_ok=True)
        root, ext = os.path.splitext(os.path.basename(save_path))
        fallback = os.path.join(
            fallback_dir, f'{root}_{int(timeit.default_timer() * 1000)}{ext}'
        )
        fig.savefig(fallback, dpi=200, bbox_inches='tight')
        return fallback


def plot_comparison_pf(mcmc_pf, psl_pf, reference_pf, metrics_by_method,
                       test_ins, n_obj, save_path=None,
                       mcmc_label='STCH-MCMC', psl_label='STCH-PSL'):
    fig = plt.figure(figsize=(11, 5))

    if n_obj == 3:
        ax = fig.add_subplot(121, projection='3d')
        if reference_pf is not None:
            ax.scatter(reference_pf[:, 0], reference_pf[:, 1], reference_pf[:, 2],
                       c='0.65', s=10, alpha=0.25, label='Reference')
        if mcmc_pf is not None and len(mcmc_pf) > 0:
            ax.scatter(mcmc_pf[:, 0], mcmc_pf[:, 1], mcmc_pf[:, 2],
                       c='tomato', s=28, alpha=0.75, label=mcmc_label)
        if psl_pf is not None and len(psl_pf) > 0:
            ax.scatter(psl_pf[:, 0], psl_pf[:, 1], psl_pf[:, 2],
                       c='royalblue', s=28, alpha=0.75, label=psl_label)
        ax.set_xlabel(r'$f_1(x)$', size=12)
        ax.set_ylabel(r'$f_2(x)$', size=12)
        ax.set_zlabel(r'$f_3(x)$', size=12)
    else:
        ax = fig.add_subplot(121)
        if reference_pf is not None:
            ax.plot(reference_pf[:, 0], reference_pf[:, 1],
                    c='0.45', lw=1.5, alpha=0.7, label='Reference')
        if mcmc_pf is not None and len(mcmc_pf) > 0:
            ax.scatter(mcmc_pf[:, 0], mcmc_pf[:, 1],
                       c='tomato', alpha=0.7, s=24, label=mcmc_label)
        if psl_pf is not None and len(psl_pf) > 0:
            ax.scatter(psl_pf[:, 0], psl_pf[:, 1],
                       c='royalblue', alpha=0.7, s=24, label=psl_label)
        ax.set_xlabel(r'$f_1(x)$', size=16)
        ax.set_ylabel(r'$f_2(x)$', size=16)
        ax.grid(alpha=0.35)

    ax.set_title(f'{test_ins} Pareto fronts')
    ax.legend(fontsize=11)

    table_ax = fig.add_subplot(122)
    table_ax.axis('off')
    rows = []
    for method, metrics in metrics_by_method.items():
        rows.append([
            method,
            _metric_text(metrics.get('hypervolume')),
            _metric_text(metrics.get('igd')),
            _metric_text(metrics.get('diversity')),
        ])
    table = table_ax.table(
        cellText=rows,
        colLabels=['Method', 'HV', 'IGD', 'Diversity'],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.6)
    table_ax.set_title('Metrics', pad=12)

    fig.tight_layout()
    saved_path = None
    if save_path:
        saved_path = save_figure(fig, save_path)
    plt.close(fig)
    return saved_path


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

        reference_pf = get_reference_front(test_ins, n_obj, ideal, nadir)
        ref_point = np.array([1.1] * n_obj)

        # Build and run MCMC sampler
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

        mcmc_metrics, mcmc_pf = sampler.compute_metrics(
            ref_point=ref_point, reference_pf=reference_pf
        )
        print(f"\n  STCH-MCMC time: {stop - start:.2f}s")

        # Run original PSL-STCH baseline in the same execution
        start = timeit.default_timer()
        _, psl_F = run_psl_stch(
            problem=problem,
            n_steps=PSL_N_STEPS,
            n_pref_update=PSL_N_PREF_UPDATE,
            ideal_point=ideal,
            nadir_point=nadir,
            device=device,
        )
        stop = timeit.default_timer()
        psl_metrics, psl_pf = compute_front_metrics(
            psl_F, n_obj, ref_point=ref_point, reference_pf=reference_pf
        )
        print(f"  STCH-PSL  time: {stop - start:.2f}s")

        metrics_by_method = {
            'STCH-MCMC': mcmc_metrics,
            'STCH-PSL': psl_metrics,
        }

        print("\n  Comparison metrics")
        print("  Method        HV        IGD       Diversity")
        for method_name, metrics in metrics_by_method.items():
            print(
                f"  {method_name:10s}  "
                f"{_metric_text(metrics.get('hypervolume')):>8s}  "
                f"{_metric_text(metrics.get('igd')):>8s}  "
                f"{_metric_text(metrics.get('diversity')):>9s}"
            )
        print(f"  STCH-MCMC uncertainty: {np.array(mcmc_metrics['uncertainty'])}")
        print(f"  STCH-MCMC archive size: {mcmc_metrics['archive_size']}")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(repo_root, 'comparison_plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{test_ins}_comparison.png')
        saved_path = plot_comparison_pf(
            mcmc_pf=mcmc_pf,
            psl_pf=psl_pf,
            reference_pf=reference_pf,
            metrics_by_method=metrics_by_method,
            test_ins=test_ins,
            n_obj=n_obj,
            save_path=save_path,
        )
        print(f"  Comparison plot saved to: {saved_path}")

        print("\n" + "*" * 70)
