"""
Conditional STCH posterior optimization per preference.

Each preference vector lambda is fixed, and a separate MALA chain samples

    p_T(x | lambda) is proportional to exp(-L_mu(f(x), lambda) / T) p(x)

on the box-constrained solution space. The sampler uses the unconstrained
parameterization x = sigmoid(y) and a Metropolis-Hastings correction, so each
conditional chain targets the exact fixed-temperature Gibbs posterior for its
assigned preference.

For optimizer benchmarking, the posterior is annealed and the reported front is
the per-preference elite/MAP front after deterministic STCH polishing. Posterior
samples are still kept for uncertainty diagnostics, but they are not used as the
primary Pareto-front output.
"""

import os
import timeit

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

INS_LIST = [
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
    're21', 're24', 're33', 're36', 're37',
]

N_PREFERENCES_2D = 200
N_PREFERENCES_3D = 210
CHAINS_PER_PREFERENCE_2D = 3
CHAINS_PER_PREFERENCE_3D = 2

N_ITER = 800
BURN_IN_FRACTION = 0.5
THIN = 10

MU_START = 0.08
MU_END = 0.006
T_START = 0.02
T_END = 0.0005
RHO = 0.0

MALA_STEP_SIZE = 2e-4
MALA_STEP_MIN = 1e-6
MALA_STEP_MAX = 5e-3
ADAPT_WINDOW = 50
TARGET_ACCEPT_LOW = 0.55
TARGET_ACCEPT_HIGH = 0.75

# Warm-start is only initialization; it is not part of the MALA transition.
WARMSTART_STEPS = 300
WARMSTART_LR = 5e-3
POLISH_STEPS = 500
POLISH_LR = 1e-2

PSL_N_STEPS = 2000
PSL_N_PREF_UPDATE = 10

DEVICE = 'cpu'
SEED = 0


def geometric_schedule(t, n_iter, start, end):
    if n_iter <= 1:
        return end
    progress = (t - 1) / (n_iter - 1)
    return start * (end / start) ** progress


def get_conditional_config(n_obj):
    if n_obj == 2:
        return N_PREFERENCES_2D, CHAINS_PER_PREFERENCE_2D
    if n_obj == 3:
        return N_PREFERENCES_3D, CHAINS_PER_PREFERENCE_3D
    raise ValueError(f"No conditional config for {n_obj} objectives.")


class ConditionalSTCHMALA:
    """Independent annealed MALA chains for fixed conditional preferences."""

    def __init__(self, problem, lambdas, n_iter=N_ITER,
                 burn_in_fraction=BURN_IN_FRACTION, thin=THIN,
                 mu_start=MU_START, mu_end=MU_END,
                 temperature_start=T_START, temperature_end=T_END,
                 rho=RHO, step_size=MALA_STEP_SIZE,
                 chains_per_preference=1, normalize=True,
                 ideal_point=None, nadir_point=None, device=DEVICE):
        self.problem = problem
        self.base_lambdas_np = np.asarray(lambdas, dtype=np.float64)
        self.n_preferences = self.base_lambdas_np.shape[0]
        self.chains_per_preference = chains_per_preference
        self.lambdas_np = np.repeat(
            self.base_lambdas_np, self.chains_per_preference, axis=0
        )
        self.n_chains = self.lambdas_np.shape[0]
        self.n_iter = n_iter
        self.burn_in_start = int(n_iter * burn_in_fraction) + 1
        self.thin = thin
        self.mu_start = mu_start
        self.mu_end = mu_end
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.current_mu = mu_start
        self.current_temperature = temperature_start
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
        else:
            self.ideal_point = None
            self.nadir_point = None

        self.z_star = torch.zeros(self.n_obj, device=device, dtype=torch.float64)
        self.Lambda = torch.tensor(self.lambdas_np, device=device, dtype=torch.float64)
        self.base_Lambda = torch.tensor(
            self.base_lambdas_np, device=device, dtype=torch.float64
        )

        x_init = torch.rand(self.n_chains, self.n_dim, device=device, dtype=torch.float64)
        if WARMSTART_STEPS > 0:
            x_init = self.warm_start_x(x_init)

        self.Y = self.logit(x_init)
        self.accepted = 0
        self.proposed = 0
        self.window_accepted = 0
        self.window_proposed = 0
        self.samples_X = []
        self.samples_F = []
        self.best_X = x_init.clone()
        with torch.no_grad():
            self.best_energy, self.best_F = self.optimizer_energy(x_init, self.Lambda)
        self.polished_X = None
        self.polished_F = None

    @staticmethod
    def logit(x):
        x = torch.clamp(x, 1e-8, 1.0 - 1e-8)
        return torch.log(x) - torch.log1p(-x)

    def normalize_values(self, f_vals):
        if self.normalize:
            return (f_vals - self.ideal_point) / (self.nadir_point - self.ideal_point)
        return f_vals

    def evaluate_objectives(self, X):
        return self.normalize_values(self.problem.evaluate(X))

    def stch_energy_from_f(self, f_vals, lambdas, mu):
        delta = f_vals - self.z_star
        smooth = mu * torch.logsumexp(lambdas * delta / mu, dim=1)
        augmented = self.rho * torch.sum(lambdas * delta, dim=1)
        return smooth + augmented

    def energy(self, X, lambdas=None, mu=None):
        if lambdas is None:
            lambdas = self.Lambda
        if mu is None:
            mu = self.current_mu
        f_vals = self.evaluate_objectives(X)
        return self.stch_energy_from_f(f_vals, lambdas, mu), f_vals

    def optimizer_energy(self, X, lambdas):
        return self.energy(X, lambdas=lambdas, mu=self.mu_end)

    def warm_start_x(self, x_init):
        x_param = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_param], lr=WARMSTART_LR)
        for _ in range(WARMSTART_STEPS):
            optimizer.zero_grad()
            x_box = torch.clamp(x_param, 1e-6, 1.0 - 1e-6)
            energy, _ = self.energy(x_box, mu=self.mu_start)
            loss = torch.sum(energy)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_param.clamp_(1e-6, 1.0 - 1e-6)
        return x_param.detach()

    def log_posterior(self, Y):
        X = torch.sigmoid(Y)
        energy, f_vals = self.energy(X)
        log_jac_x = (F.logsigmoid(Y) + F.logsigmoid(-Y)).sum(dim=1)
        log_prob = -energy / self.current_temperature + log_jac_x
        return log_prob, X, f_vals

    def logp_and_grad(self, Y):
        Y_req = Y.detach().clone().requires_grad_(True)
        logp, X, f_vals = self.log_posterior(Y_req)
        grad_Y = torch.autograd.grad(logp.sum(), Y_req, create_graph=False)[0]
        return logp.detach(), grad_Y.detach(), X.detach(), f_vals.detach()

    @staticmethod
    def log_gaussian_kernel(value, mean, step_size):
        diff = value - mean
        return -0.5 * diff.reshape(diff.shape[0], -1).pow(2).sum(dim=1) / step_size

    def mala_step(self):
        step = self.step_size
        logp, grad_Y, _, _ = self.logp_and_grad(self.Y)

        mean_Y = self.Y + 0.5 * step * grad_Y
        proposal_Y = mean_Y + np.sqrt(step) * torch.randn_like(self.Y)

        prop_logp, prop_grad_Y, X_prop, F_prop = self.logp_and_grad(proposal_Y)
        reverse_mean_Y = proposal_Y + 0.5 * step * prop_grad_Y

        log_q_forward = self.log_gaussian_kernel(proposal_Y, mean_Y, step)
        log_q_reverse = self.log_gaussian_kernel(self.Y, reverse_mean_Y, step)
        log_accept_ratio = prop_logp + log_q_reverse - logp - log_q_forward
        accept = torch.log(torch.rand(self.n_chains, device=self.device)) < torch.clamp(
            log_accept_ratio, max=0.0
        )

        self.Y = torch.where(accept[:, None], proposal_Y, self.Y).detach()
        accepted = int(accept.sum().item())
        self.accepted += accepted
        self.proposed += self.n_chains
        self.window_accepted += accepted
        self.window_proposed += self.n_chains

        with torch.no_grad():
            _, X, f_vals = self.log_posterior(self.Y)
            final_energy = self.stch_energy_from_f(f_vals, self.Lambda, self.mu_end)
            self.update_best(X, f_vals, final_energy)
        return X, f_vals

    def update_best(self, X, f_vals, final_energy):
        finite = torch.isfinite(final_energy) & torch.isfinite(f_vals).all(dim=1)
        improved = finite & (final_energy < self.best_energy)
        if torch.any(improved):
            self.best_energy[improved] = final_energy[improved]
            self.best_X[improved] = X[improved]
            self.best_F[improved] = f_vals[improved]

    def adapt_step_size(self):
        if self.window_proposed == 0:
            return
        rate = self.window_accepted / self.window_proposed
        if rate > TARGET_ACCEPT_HIGH:
            self.step_size = min(self.step_size * 1.5, MALA_STEP_MAX)
        elif rate < TARGET_ACCEPT_LOW:
            self.step_size = max(self.step_size * 0.7, MALA_STEP_MIN)
        self.window_accepted = 0
        self.window_proposed = 0

    def run(self):
        for t in range(1, self.n_iter + 1):
            self.current_mu = geometric_schedule(t, self.n_iter, self.mu_start, self.mu_end)
            self.current_temperature = geometric_schedule(
                t, self.n_iter, self.temperature_start, self.temperature_end
            )
            X, f_vals = self.mala_step()
            if t >= self.burn_in_start and (t - self.burn_in_start) % self.thin == 0:
                self.samples_X.append(X.cpu().clone().numpy())
                self.samples_F.append(f_vals.cpu().clone().numpy())
            if t <= self.burn_in_start and t % ADAPT_WINDOW == 0:
                self.adapt_step_size()
            if t == 1 or t % 500 == 0:
                print(
                    f"  Iter {t:5d}/{self.n_iter} | "
                    f"mu={self.current_mu:.4g} | "
                    f"T={self.current_temperature:.4g} | "
                    f"step={self.step_size:.2e} | "
                    f"accept={self.acceptance_rate:.3f}"
                )
        self.polish_best()
        print("  Conditional MALA optimization complete.")

    @property
    def acceptance_rate(self):
        if self.proposed == 0:
            return 0.0
        return self.accepted / self.proposed

    def get_collected_samples(self):
        if len(self.samples_F) == 0:
            return None, None, None
        X_all = np.concatenate(self.samples_X, axis=0)
        F_all = np.concatenate(self.samples_F, axis=0)
        lambda_repeated = np.tile(self.lambdas_np, (len(self.samples_F), 1))
        return X_all, lambda_repeated, F_all

    def select_best_by_preference(self):
        X = self.best_X.detach().cpu().numpy().reshape(
            self.n_preferences, self.chains_per_preference, self.n_dim
        )
        F_best = self.best_F.detach().cpu().numpy().reshape(
            self.n_preferences, self.chains_per_preference, self.n_obj
        )
        energy = self.best_energy.detach().cpu().numpy().reshape(
            self.n_preferences, self.chains_per_preference
        )
        best_idx = np.nanargmin(energy, axis=1)
        rows = np.arange(self.n_preferences)
        return X[rows, best_idx], F_best[rows, best_idx]

    def polish_best(self):
        x_best, _ = self.select_best_by_preference()
        y_param = self.logit(
            torch.tensor(x_best, device=self.device, dtype=torch.float64)
        ).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([y_param], lr=POLISH_LR)

        best_X = torch.sigmoid(y_param.detach())
        with torch.no_grad():
            best_energy, best_F = self.optimizer_energy(best_X, self.base_Lambda)

        for _ in range(POLISH_STEPS):
            optimizer.zero_grad()
            X = torch.sigmoid(y_param)
            energy, _ = self.optimizer_energy(X, self.base_Lambda)
            loss = torch.sum(energy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([y_param], max_norm=100.0)
            optimizer.step()

            with torch.no_grad():
                X_eval = torch.sigmoid(y_param)
                energy_eval, F_eval = self.optimizer_energy(X_eval, self.base_Lambda)
                improved = torch.isfinite(energy_eval) & (energy_eval < best_energy)
                if torch.any(improved):
                    best_energy[improved] = energy_eval[improved]
                    best_X[improved] = X_eval[improved]
                    best_F[improved] = F_eval[improved]

        self.polished_X = best_X.detach().cpu().numpy()
        self.polished_F = best_F.detach().cpu().numpy()
        return self.polished_X, self.polished_F

    def compute_metrics(self, ref_point=None, reference_pf=None, summary='optimizer'):
        X_all, Lambda_all, F_all = self.get_collected_samples()

        if summary == 'optimizer':
            if self.polished_F is None:
                self.polish_best()
            metric_F = self.polished_F
        elif F_all is None:
            return {}, None
        elif summary == 'mean':
            F_by_draw = F_all.reshape(len(self.samples_F), self.n_chains, self.n_obj)
            metric_F = F_by_draw.mean(axis=0)
        elif summary == 'best':
            _, metric_F = self.select_best_by_preference()
        elif summary == 'samples':
            metric_F = F_all
        else:
            raise ValueError(f"Unknown summary mode: {summary}")

        metrics, pf = compute_front_metrics(
            metric_F, self.n_obj, ref_point=ref_point, reference_pf=reference_pf
        )
        if F_all is not None:
            metrics['uncertainty'] = np.std(F_all, axis=0).tolist()
        else:
            metrics['uncertainty'] = [0.0] * self.n_obj
        metrics['acceptance_rate'] = self.acceptance_rate
        metrics['final_step_size'] = self.step_size
        metrics['front_size'] = 0 if pf is None else len(pf)
        return metrics, pf


if __name__ == '__main__':
    setup_plot_style()
    summary_rows = []

    for problem_idx, test_ins in enumerate(INS_LIST):
        np.random.seed(SEED + problem_idx)
        torch.manual_seed(SEED + problem_idx)

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
        n_preferences, chains_per_preference = get_conditional_config(n_obj)
        lambdas = get_stratified_preferences(n_preferences, n_obj)

        sampler = ConditionalSTCHMALA(
            problem=problem,
            lambdas=lambdas,
            n_iter=N_ITER,
            burn_in_fraction=BURN_IN_FRACTION,
            thin=THIN,
            mu_start=MU_START,
            mu_end=MU_END,
            temperature_start=T_START,
            temperature_end=T_END,
            rho=RHO,
            step_size=MALA_STEP_SIZE,
            chains_per_preference=chains_per_preference,
            normalize=True,
            ideal_point=ideal,
            nadir_point=nadir,
            device=DEVICE,
        )

        start = timeit.default_timer()
        sampler.run()
        stop = timeit.default_timer()
        cond_metrics, cond_pf = sampler.compute_metrics(
            ref_point=ref_point, reference_pf=reference_pf
        )
        print(f"\n  Conditional STCH-MALA time: {stop - start:.2f}s")

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
        print(f"  STCH-PSL time: {stop - start:.2f}s")

        metrics_by_method = {
            'Cond-MALA': cond_metrics,
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
        print(f"  Cond-MALA acceptance: {cond_metrics['acceptance_rate']:.3f}")
        print(f"  Cond-MALA final step size: {cond_metrics['final_step_size']:.2e}")
        print(f"  Cond-MALA uncertainty: {np.array(cond_metrics['uncertainty'])}")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(repo_root, 'comparison_plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{test_ins}_conditional_mala_comparison.png')
        saved_path = plot_comparison_pf(
            mcmc_pf=cond_pf,
            psl_pf=psl_pf,
            reference_pf=reference_pf,
            metrics_by_method=metrics_by_method,
            test_ins=test_ins,
            n_obj=n_obj,
            save_path=save_path,
            mcmc_label='Cond-MALA',
            psl_label='STCH-PSL',
        )
        print(f"  Comparison plot saved to: {saved_path}")

        cond_wins_hv = (
            cond_metrics.get('hypervolume', np.nan) > psl_metrics.get('hypervolume', np.nan)
        )
        cond_wins_igd = (
            cond_metrics.get('igd', np.inf) < psl_metrics.get('igd', np.inf)
        )
        summary_rows.append([
            test_ins,
            n_obj,
            n_preferences,
            chains_per_preference,
            cond_metrics.get('hypervolume', np.nan),
            cond_metrics.get('igd', np.nan),
            cond_metrics.get('diversity', np.nan),
            cond_metrics.get('acceptance_rate', np.nan),
            cond_metrics.get('front_size', np.nan),
            psl_metrics.get('hypervolume', np.nan),
            psl_metrics.get('igd', np.nan),
            psl_metrics.get('diversity', np.nan),
            len(psl_pf) if psl_pf is not None else 0,
            cond_wins_hv,
            cond_wins_igd,
        ])

        print("\n" + "*" * 70)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(repo_root, 'comparison_plots')
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'conditional_all_results.csv')
    header = (
        'problem,n_obj,n_preferences,chains_per_preference,'
        'cond_hv,cond_igd,cond_diversity,cond_acceptance,cond_front_size,'
        'psl_hv,psl_igd,psl_diversity,psl_front_size,cond_wins_hv,cond_wins_igd'
    )
    np.savetxt(
        summary_path,
        np.asarray(summary_rows, dtype=object),
        fmt='%s',
        delimiter=',',
        header=header,
        comments='',
    )
    print(f"\nSummary saved to: {summary_path}")
