# AGENT.md — Smooth Tchebycheff Posterior Sampling for Multi-Objective Optimization

---

## Objective

Build a Bayesian MOO framework that:
1. Maintains a **posterior distribution over the Pareto front** — not a point estimate
2. Uses **Smooth Tchebycheff** as the energy function defining the posterior
3. Samples via **joint MCMC over (x, λ)** — solution space and preference space simultaneously

---

## Motivation

Classical MOO (NSGA-II, MOEA/D) returns a fixed Pareto front — no uncertainty.
GP-based Bayesian MOO (qEHVI, PESMO) maintains a posterior but cannot scale to neural objectives.

This method fixes both:
- Replace the GP surrogate with **direct Langevin MCMC in solution space** — only needs ∇f_k(x), computable via backprop
- Use **Smooth Tchebycheff** as the energy — differentiable everywhere, recovers non-convex Pareto fronts, adapts gradient to the worst-violating objective automatically

---

## Probabilistic Model

```
p(λ)              — prior over preference vectors  (Dirichlet or Uniform on simplex)

p(x | λ)  ∝  exp( −(1/T) · L_μ(f(x), λ) )       — preference-conditioned solution dist.

p(x, λ)   =  p(x | λ) · p(λ)                      — joint distribution

p(x)      =  ∫ p(x | λ) p(λ) dλ                   — marginal = posterior over Pareto front
```

As T → 0, mass concentrates on the Pareto-optimal set.
At finite T, the support gives credible regions around the Pareto front.

---

## Energy Function: Smooth Tchebycheff

```
L_μ(f(x); λ, z*)  =  (1/μ) · log Σ_k  exp( μ · λ_k · (f_k(x) − z*_k) )
```

Parameters:
- `λ`    — preference vector on simplex Δ^K, encodes trade-off priority
- `z*`   — utopia point, best known value per objective (updated during sampling)
- `μ`    — smoothing parameter
  - μ → 0    :  reduces to weighted sum scalarization
  - μ finite  :  smooth Tchebycheff (differentiable everywhere)
  - μ → ∞    :  recovers hard Tchebycheff (max over k, non-differentiable)

---

## Score Function

```
∇_x log p(x | λ)  =  −(1/T) · Σ_k  w_k(x, λ) · λ_k · ∇_x f_k(x)

where:
    w_k(x, λ)  =  softmax_k( μ · λ_k · (f_k(x) − z*_k) )     adaptive weights
```

Interpretation of w_k:
- High w_k  →  objective k is currently the most violated
- Gradient mass automatically focuses on the bottleneck objective
- No manual scheduling needed

```
∇_λ log p(λ | x)  =  −(1/T) · w(x, λ) ⊙ (f(x) − z*)   +   (α − 1) / λ
                       ───────────────────────────────────     ────────────
                            likelihood gradient                prior gradient (Dirichlet)
```

---

## Augmented Tchebycheff (Optional)

Add a linear term to better recover non-convex Pareto fronts:

```
L_aug(f(x); λ, z*)  =  L_μ(f(x); λ, z*)   +   ρ · Σ_k  λ_k · (f_k(x) − z*_k)

Effect on weights:
    w_k (augmented)  =  w_k (smooth)  +  ρ
```

The ρ term makes the energy strictly convex. Typical values: ρ = 0.01 – 0.1

---

## Annealing Schedule for (μ, T)

```
Phase 1 — Exploration   (first half of iterations)
    μ : small → medium      landscape is smooth, easy to explore
    T : large → medium      distribution is diffuse
    Rule: keep μ · T = const   (preserves effective temperature)

Phase 2 — Exploitation  (second half of iterations)
    μ : fix at large value  approach hard Tchebycheff
    T : decrease → 0        concentrate mass on Pareto front
```

Rule of thumb:
- μ_end  = 10 – 50 depending on K
- T_end  = 1e-2 – 1e-3
- ε_x  <  ε_λ  (solution space usually higher-dimensional)

---

## Main Algorithm

```
ALGORITHM: Smooth-Tchebycheff Joint MCMC

INPUT:
    f           — objective functions f_1, ..., f_K
    N           — number of particles
    n_iter      — total MCMC iterations
    μ_schedule  — annealing schedule for smoothing parameter
    T_schedule  — annealing schedule for temperature
    ε_x         — step size for x-update
    ε_λ         — step size for λ-update
    α           — Dirichlet prior concentration

INITIALIZE:
    X   ← N random particles in solution space
    Λ   ← N uniform vectors on simplex Δ^K
    z*  ← +∞  for each objective

FOR t = 1 to n_iter:

    μ ← μ_schedule(t)
    T ← T_schedule(t)

    ── Evaluate objectives and gradients ──────────────────
    FOR each particle i:
        f_i ← evaluate  f(x_i)                  [K values]
        G_i ← compute   ∇_x f_k(x_i)  ∀k        [K gradients]

    ── Update utopia point ────────────────────────────────
    z*_k ← min over all particles of f_k^(i)    for each k

    ── Update each particle ───────────────────────────────
    FOR each particle i:

        // Compute adaptive weights
        a_k   ←  μ · λ_k^(i) · (f_k^(i) − z*_k)    for each k
        w_k   ←  softmax(a)_k

        // x-update via Langevin
        score_x  ←  −(1/T) · Σ_k  w_k · λ_k^(i) · G_k^(i)
        x^(i)    ←  x^(i)  +  (ε_x / 2) · score_x  +  √ε_x · Gaussian noise

        // λ-update via Projected Langevin on simplex
        score_λ  ←  −(1/T) · w^(i) ⊙ (f^(i) − z*)  +  (α−1) / λ^(i)
        λ^(i)    ←  project_simplex( λ^(i)  +  (ε_λ / 2) · score_λ  +  √ε_λ · noise )

    ── Collect samples after burn-in ──────────────────────
    IF t  >  n_iter / 2:
        save current (X, Λ)

OUTPUT:
    {(x^(i), λ^(i))}  ≈  samples from joint  p(x, λ)
    {x^(i)}            ≈  posterior over Pareto front
```

---

## Simplex Projection Subroutine

```
SUBROUTINE: project_simplex(v)

INPUT:  v — unconstrained vector of length K

Sort v in descending order → u

Find largest index ρ such that:
    u_ρ  >  (1/ρ) · ( Σ_{j=1}^{ρ} u_j  −  1 )

Compute threshold:
    θ  =  (1/ρ) · ( Σ_{j=1}^{ρ} u_j  −  1 )

OUTPUT:
    λ_k  =  max( v_k − θ,  0 )    for each k
```

---

## Evaluation Metrics

```
Hypervolume       — volume dominated by the empirical Pareto front
                    higher = better coverage and quality

Diversity         — average pairwise distance among Pareto particles
                    higher = better spread across the front

Uncertainty per k — standard deviation of f_k across Pareto particles
                    high = front still uncertain in objective k
                    → prioritize this region for next expensive evaluation
```

---

## Pitfalls and Fixes

```
PROBLEM                          CAUSE                     FIX
──────────────────────────────────────────────────────────────────────
Particles collapse to one point  T decreases too fast      Slow down T annealing
Pareto front not fully covered   ε_λ too small             Increase λ step size
z* overestimated                 Bad initialization        Init z* = +∞, update gradually
Gradient explosion               μ increases too fast      Cap μ, clip gradients
λ collapses to simplex corner    α too small               Increase α  (e.g. α = 2.0)
```

---

## References

- Smooth Tchebycheff: Liu et al. (2024)
- MOEA/D decomposition: Zhang & Li (2007)
- MOO-SVGD: Liu et al. (2021)
- Pareto HyperNetworks: Navon et al. (2021)
- PESMO: Hernández-Lobato et al. (2016)
- qNEHVI: Daulton et al. (2021)
- Unadjusted Langevin Algorithm: Welling & Teh (2011)