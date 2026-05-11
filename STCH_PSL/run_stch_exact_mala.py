"""
Metropolis-adjusted Langevin sampler for the STCH posterior.

This script samples a fixed target posterior with a valid Metropolis-Hastings
correction. It is "exact" in the MCMC sense that, for fixed mu, temperature,
utopia point, and priors, the chain has the requested posterior as its stationary
distribution. It deliberately does not use annealing, projection, clipping, or a
post-hoc optimizer inside the sampling transition.
"""

import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from problem import get_problem
from run_stch_mcmc import (
    compute_front_metrics,
    get_reference_front,
    get_stratified_preferences,
    plot_comparison_pf,
    run_psl_stch,
    setup_plot_style,
)


# ------------------------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------------------------

INS_LIST = ['f1']

N_PARTICLES = 64
N_ITER = 2000
BURN_IN_FRACTION = 0.5
THIN = 5

# Fixed posterior target: p(x, lambda) proportional to exp(-L_mu / T) p(lambda).
POSTERIOR_MU = 0.01
POSTERIOR_T = 0.005
DIRICHLET_ALPHA = 1.0
RHO = 0.0

# MALA step size in unconstrained coordinates.
MALA_STEP_SIZE = 2e-4

# Warm-start only affects initialization; it is not part of the Markov kernel.
WARMSTART_STEPS = 500
WARMSTART_LR = 5e-3

PSL_N_STEPS = 2000
PSL_N_PREF_UPDATE = 10

DEVICE = 'cpu'


# ------------------------------------------------------------------------------
# Exact fixed-target MALA
# ------------------------------------------------------------------------------


class ExactSTCHMALA:
    """MALA sampler for the fixed STCH posterior on box x and simplex lambda."""

    def __init__(self, problem, n_particles=N_PARTICLES, n_iter=N_ITER,
                 burn_in_fraction=BURN_IN_FRACTION, thin=THIN,
                 mu=POSTERIOR_MU, temperature=POSTERIOR_T,
                 alpha=DIRICHLET_ALPHA, rho=RHO,
                 step_size=MALA_STEP_SIZE, normalize=True,
                 ideal_point=None, nadir_point=None, device=DEVICE):
        self.problem = problem
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.burn_in_start = int(n_iter * burn_in_fraction) + 1
        self.thin = thin
        self.mu = mu
        self.temperature = temperature
        self.alpha = alpha
        self.rho = rho
        self.step_size = step_size
        self.normalize = normalize
        self.device = device

        self.n_dim = problem.n_dim
        self.n_obj = problem.n_obj

        if normalize:
            self.ideal_point = torch.tensor(
                np.zeros(self.n_obj) if ideal_point is None else ideal_point,
                device=device,
                dtype=torch.float64,
            )
            self.nadir_point = torch.tensor(
                np.ones(self.n_obj) if nadir_point is None else nadir_point,
                device=device,
                dtype=torch.float64,
            )
            self.z_star = torch.zeros(self.n_obj, device=device, dtype=torch.float64)
        else:
            self.ideal_point = None
            self.nadir_point = None
            self.z_star = torch.zeros(self.n_obj, device=device, dtype=torch.float64)

        lambdas = get_stratified_preferences(n_particles, self.n_obj)
        x_init = torch.rand(n_particles, self.n_dim, device=device, dtype=torch.float64)
        lambda_init = torch.tensor(lambdas, device=device, dtype=torch.float64)

        if WARMSTART_STEPS > 0:
            x_init = self.warm_start_x(x_init, lambda_init)

        self.Y = self.logit(x_init)
        self.Theta = self.theta_from_lambda(lambda_init)

        self.accepted = 0
        self.proposed = 0
        self.samples_X = []
        self.samples_Lambda = []
        self.samples_F = []

    @staticmethod
    def logit(x):
        x = torch.clamp(x, 1e-8, 1.0 - 1e-8)
        return torch.log(x) - torch.log1p(-x)

    @staticmethod
    def theta_from_lambda(lam):
        lam = torch.clamp(lam, 1e-12, 1.0)
        lam = lam / lam.sum(dim=1, keepdim=True)
        return torch.log(lam[:, :-1]) - torch.log(lam[:, -1:])

    def lambda_from_theta(self, theta):
        zeros = torch.zeros(theta.shape[0], 1, device=self.device, dtype=torch.float64)
        logits = torch.cat([theta, zeros], dim=1)
        return torch.softmax(logits, dim=1)

    def normalize_values(self, f_vals):
        if self.normalize:
            return (f_vals - self.ideal_point) / (self.nadir_point - self.ideal_point)
        return f_vals

    def evaluate_objectives(self, X):
        return self.normalize_values(self.problem.evaluate(X))

    def energy(self, X, Lambda):
        f_vals = self.evaluate_objectives(X)
        delta = f_vals - self.z_star
        smooth = self.mu * torch.logsumexp(Lambda * delta / self.mu, dim=1)
        augmented = self.rho * torch.sum(Lambda * delta, dim=1)
        return smooth + augmented, f_vals

    def warm_start_x(self, x_init, lambda_init):
        """Find a high-density starting point without changing the MCMC target."""
        x_param = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_param], lr=WARMSTART_LR)
        for _ in range(WARMSTART_STEPS):
            optimizer.zero_grad()
            x_box = torch.clamp(x_param, 1e-6, 1.0 - 1e-6)
            energy, _ = self.energy(x_box, lambda_init)
            loss = torch.sum(energy)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_param.clamp_(1e-6, 1.0 - 1e-6)
        return x_param.detach()

    def log_posterior(self, Y, Theta):
        """Log density in unconstrained coordinates, including Jacobians."""
        X = torch.sigmoid(Y)
        Lambda = self.lambda_from_theta(Theta)
        energy, f_vals = self.energy(X, Lambda)

        log_jac_x = (F.logsigmoid(Y) + F.logsigmoid(-Y)).sum(dim=1)
        log_lambda = torch.log(torch.clamp(Lambda, min=1e-30))

        # Dirichlet(alpha) prior plus additive-log-ratio Jacobian prod(lambda_k).
        log_lambda_density = self.alpha * log_lambda.sum(dim=1)
        log_prob = -energy / self.temperature + log_jac_x + log_lambda_density
        return log_prob, X, Lambda, f_vals

    def logp_and_grad(self, Y, Theta):
        Y_req = Y.detach().clone().requires_grad_(True)
        Theta_req = Theta.detach().clone().requires_grad_(True)
        logp, X, Lambda, f_vals = self.log_posterior(Y_req, Theta_req)
        grad_Y, grad_Theta = torch.autograd.grad(
            logp.sum(), [Y_req, Theta_req], create_graph=False
        )
        return (
            logp.detach(),
            grad_Y.detach(),
            grad_Theta.detach(),
            X.detach(),
            Lambda.detach(),
            f_vals.detach(),
        )

    @staticmethod
    def log_gaussian_kernel(value, mean, step_size):
        diff = value - mean
        return -0.5 * diff.reshape(diff.shape[0], -1).pow(2).sum(dim=1) / step_size

    def mala_step(self):
        step = self.step_size
        logp, grad_Y, grad_Theta, _, _, _ = self.logp_and_grad(self.Y, self.Theta)

        mean_Y = self.Y + 0.5 * step * grad_Y
        mean_Theta = self.Theta + 0.5 * step * grad_Theta
        proposal_Y = mean_Y + np.sqrt(step) * torch.randn_like(self.Y)
        proposal_Theta = mean_Theta + np.sqrt(step) * torch.randn_like(self.Theta)

        prop_logp, prop_grad_Y, prop_grad_Theta, X_prop, Lambda_prop, F_prop = (
            self.logp_and_grad(proposal_Y, proposal_Theta)
        )

        reverse_mean_Y = proposal_Y + 0.5 * step * prop_grad_Y
        reverse_mean_Theta = proposal_Theta + 0.5 * step * prop_grad_Theta

        log_q_forward = (
            self.log_gaussian_kernel(proposal_Y, mean_Y, step)
            + self.log_gaussian_kernel(proposal_Theta, mean_Theta, step)
        )
        log_q_reverse = (
            self.log_gaussian_kernel(self.Y, reverse_mean_Y, step)
            + self.log_gaussian_kernel(self.Theta, reverse_mean_Theta, step)
        )

        log_accept_ratio = prop_logp + log_q_reverse - logp - log_q_forward
        accept = torch.log(torch.rand(self.n_particles, device=self.device)) < torch.clamp(
            log_accept_ratio, max=0.0
        )

        self.Y = torch.where(accept[:, None], proposal_Y, self.Y).detach()
        self.Theta = torch.where(accept[:, None], proposal_Theta, self.Theta).detach()
        self.accepted += int(accept.sum().item())
        self.proposed += self.n_particles

        with torch.no_grad():
            _, X, Lambda, f_vals = self.log_posterior(self.Y, self.Theta)
        return X, Lambda, f_vals

    def run(self):
        for t in range(1, self.n_iter + 1):
            X, Lambda, f_vals = self.mala_step()
            if t >= self.burn_in_start and (t - self.burn_in_start) % self.thin == 0:
                self.samples_X.append(X.cpu().clone().numpy())
                self.samples_Lambda.append(Lambda.cpu().clone().numpy())
                self.samples_F.append(f_vals.cpu().clone().numpy())
            if t == 1 or t % 500 == 0:
                print(
                    f"  Iter {t:5d}/{self.n_iter} | "
                    f"accept={self.acceptance_rate:.3f}"
                )
        print("  Exact MALA sampling complete.")

    @property
    def acceptance_rate(self):
        if self.proposed == 0:
            return 0.0
        return self.accepted / self.proposed

    def get_collected_samples(self):
        if len(self.samples_F) == 0:
            return None, None, None
        return (
            np.concatenate(self.samples_X, axis=0),
            np.concatenate(self.samples_Lambda, axis=0),
            np.concatenate(self.samples_F, axis=0),
        )

    def compute_metrics(self, ref_point=None, reference_pf=None):
        _, _, F_all = self.get_collected_samples()
        if F_all is None:
            return {}, None
        metrics, pf = compute_front_metrics(
            F_all, self.n_obj, ref_point=ref_point, reference_pf=reference_pf
        )
        metrics['uncertainty'] = np.std(F_all, axis=0).tolist()
        metrics['acceptance_rate'] = self.acceptance_rate
        return metrics, pf


# ------------------------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    setup_plot_style()

    for test_ins in INS_LIST:
        print("\n" + "=" * 70)
        print(f"Problem: {test_ins}")
        print("=" * 70)

        problem = get_problem(test_ins)
        n_obj = problem.n_obj

        if test_ins.startswith('re'):
            base = os.path.dirname(os.path.abspath(__file__))
            ideal = np.loadtxt(
                os.path.join(base, f'data/RE/ideal_nadir_points/ideal_point_{test_ins}.dat')
            )
            nadir = np.loadtxt(
                os.path.join(base, f'data/RE/ideal_nadir_points/nadir_point_{test_ins}.dat')
            )
        else:
            ideal = np.zeros(n_obj)
            nadir = np.ones(n_obj)

        reference_pf = get_reference_front(test_ins, n_obj, ideal, nadir)
        ref_point = np.array([1.1] * n_obj)

        sampler = ExactSTCHMALA(
            problem=problem,
            n_particles=N_PARTICLES,
            n_iter=N_ITER,
            burn_in_fraction=BURN_IN_FRACTION,
            thin=THIN,
            mu=POSTERIOR_MU,
            temperature=POSTERIOR_T,
            alpha=DIRICHLET_ALPHA,
            rho=RHO,
            step_size=MALA_STEP_SIZE,
            normalize=True,
            ideal_point=ideal,
            nadir_point=nadir,
            device=DEVICE,
        )

        start = timeit.default_timer()
        sampler.run()
        stop = timeit.default_timer()
        mala_metrics, mala_pf = sampler.compute_metrics(
            ref_point=ref_point, reference_pf=reference_pf
        )
        print(f"\n  STCH-MALA time: {stop - start:.2f}s")

        start = timeit.default_timer()
        _, psl_F = run_psl_stch(
            problem=problem,
            n_steps=PSL_N_STEPS,
            n_pref_update=PSL_N_PREF_UPDATE,
            ideal_point=ideal,
            nadir_point=nadir,
            device=DEVICE,
        )
        stop = timeit.default_timer()
        psl_metrics, psl_pf = compute_front_metrics(
            psl_F, n_obj, ref_point=ref_point, reference_pf=reference_pf
        )
        print(f"  STCH-PSL  time: {stop - start:.2f}s")

        metrics_by_method = {
            'STCH-MALA': mala_metrics,
            'STCH-PSL': psl_metrics,
        }

        print("\n  Comparison metrics")
        print("  Method        HV        IGD       Diversity")
        for method_name, metrics in metrics_by_method.items():
            print(
                f"  {method_name:10s}  "
                f"{metrics.get('hypervolume', np.nan):8.4f}  "
                f"{metrics.get('igd', np.nan):8.4f}  "
                f"{metrics.get('diversity', np.nan):9.4f}"
            )
        print(f"  STCH-MALA acceptance: {mala_metrics['acceptance_rate']:.3f}")
        print(f"  STCH-MALA uncertainty: {np.array(mala_metrics['uncertainty'])}")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(repo_root, 'comparison_plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{test_ins}_exact_mala_comparison.png')
        saved_path = plot_comparison_pf(
            mcmc_pf=mala_pf,
            psl_pf=psl_pf,
            reference_pf=reference_pf,
            metrics_by_method=metrics_by_method,
            test_ins=test_ins,
            n_obj=n_obj,
            save_path=save_path,
            mcmc_label='STCH-MALA',
            psl_label='STCH-PSL',
        )
        print(f"  Comparison plot saved to: {saved_path}")

        print("\n" + "*" * 70)
