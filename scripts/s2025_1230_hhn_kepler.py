#!/usr/bin/env python3
"""
Flexible trainer for MyActionAngleNetwork (AAN) on Kepler problem data,
adapted from notebooks/e2025_1125_hhn_vs_aan_kepler.ipynb and structured
following scripts/s2025_1111_my_aan.py.

Key features:
- Dataset from random (q0,p0) and variable time jumps (1..ceil(delta_t_max/dt)).
- Analytic ground-truth flow for 2D Kepler problem.
- Configurable model hyperparameters and loss weights.
- Horizon-wise RMSE (q/p) evaluation at custom test steps.
- Hamiltonian and action-angle diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import jax
from jax import lax, random, value_and_grad
import jax.numpy as jnp
import optax

from action_angle_networks.sk_models import MyActionAngleNetwork


# ------------------------- CLI helpers -------------------------

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in {"1", "y", "yes", "t", "true"}:
        return True
    if x in {"0", "n", "no", "f", "false"}:
        return False
    raise ValueError(f"Invalid boolean: {x}")


def parse_int_list(x) -> List[int]:
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    s = str(x).strip()
    if s.startswith("[") or s.startswith("("):
        import ast
        v = ast.literal_eval(s)
        return [int(i) for i in v]
    return [int(t) for t in s.replace(",", " ").split() if t]


def parse_float_list(x) -> List[float]:
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    s = str(x).strip()
    if s.startswith("[") or s.startswith("("):
        import ast
        v = ast.literal_eval(s)
        return [float(i) for i in v]
    return [float(t) for t in s.replace(",", " ").split() if t]


# ------------------------- Physics helpers (Kepler) -------------------------

@dataclass
class KeplerParams:
    m: float  # mass of the orbiting body
    k: float  # gravitational parameter (e.g., G*M)
    eps: float = 1e-8  # softening to avoid singularity at r=0


def set_kepler_params(m: float = 1.0, k: float = 1.0, eps: float = 1e-8) -> KeplerParams:
    return KeplerParams(m=m, k=k, eps=eps)


def kepler_hamiltonian(q: jnp.ndarray, p: jnp.ndarray, params: KeplerParams) -> jnp.ndarray:
    """Return the Kepler Hamiltonian H(q, p) = |p|^2/(2m) - k/|q|.

    q, p can be (..., dim). The last axis is treated as spatial dimension.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    r = jnp.linalg.norm(q, axis=-1)
    kinetic = jnp.sum(p ** 2, axis=-1) / (2.0 * params.m)
    potential = -params.k / jnp.maximum(r, params.eps)
    return kinetic + potential


def kepler_equations(q: jnp.ndarray, p: jnp.ndarray, params: KeplerParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Hamilton's equations for Kepler problem.

    Returns (dq/dt, dp/dt).
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    r = jnp.linalg.norm(q, axis=-1, keepdims=True)
    r_safe = jnp.maximum(r, params.eps)
    dqdt = p / params.m
    dpdt = -params.k * q / (r_safe ** 3)
    return dqdt, dpdt


def _cross2(a, b):
    """2D cross product (returns scalar)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _kepler_elements_from_state_2d(q0, p0, params: KeplerParams):
    """Compute Kepler orbital elements from initial state (2D).

    Returns: mu, a, e, e_hat, q_hat, M0, n
    where:
        mu: standard gravitational parameter
        a: semi-major axis
        e: eccentricity
        e_hat: eccentricity unit vector
        q_hat: perpendicular unit vector
        M0: initial mean anomaly
        n: mean motion
    """
    q0 = jnp.atleast_2d(jnp.asarray(q0, dtype=jnp.float64))
    p0 = jnp.atleast_2d(jnp.asarray(p0, dtype=jnp.float64))

    m = jnp.asarray(params.m, dtype=jnp.float64)
    k = jnp.asarray(params.k, dtype=jnp.float64)
    mu = k / m  # q¨ = -mu q/r^3

    r = jnp.linalg.norm(q0, axis=-1)
    v0 = p0 / m
    v2 = jnp.sum(v0 * v0, axis=-1)

    eps_spec = 0.5 * v2 - mu / r            # specific energy
    a = -mu / (2.0 * eps_spec)              # elliptic => a>0

    h = _cross2(q0, v0)                      # scalar angular momentum
    vxh = jnp.stack([v0[..., 1] * h, -v0[..., 0] * h], axis=-1)
    evec = vxh / mu - q0 / r[..., None]
    e = jnp.linalg.norm(evec, axis=-1)

    e_hat = evec / e[..., None]
    q_hat = jnp.stack([-e_hat[..., 1], e_hat[..., 0]], axis=-1)

    cosE0 = (1.0 - r / a) / e
    cosE0 = jnp.clip(cosE0, -1.0, 1.0)
    sinE0 = jnp.sum(q0 * v0, axis=-1) / (e * jnp.sqrt(mu * a))
    sinE0 = jnp.clip(sinE0, -1.0, 1.0)

    E0 = jnp.arctan2(sinE0, cosE0)
    M0 = E0 - e * sinE0
    n = jnp.sqrt(mu / a**3)
    return mu, a, e, e_hat, q_hat, M0, n


def _solve_kepler_E(M, e, num_iter=12):
    """Solve Kepler's equation E - e*sin(E) = M for eccentric anomaly E.

    Uses Newton-Raphson iteration.
    """
    E = M + e * jnp.sin(M) + 0.5 * (e**2) * jnp.sin(2.0 * M)
    def body(_, Ecur):
        f = Ecur - e * jnp.sin(Ecur) - M
        fp = 1.0 - e * jnp.cos(Ecur)
        return Ecur - f / fp
    return lax.fori_loop(0, num_iter, body, E)


def simulate_kepler_analytic(params: KeplerParams, num_steps: int, dt: float, q0, p0):
    """Simulate Kepler problem using analytical solution.

    Args:
        params: Kepler problem parameters
        num_steps: number of time steps
        dt: time step size
        q0: initial positions (B, 2)
        p0: initial momenta (B, 2)

    Returns:
        t: time array (num_steps+1,)
        q_traj: position trajectory (num_steps+1, B, 2)
        p_traj: momentum trajectory (num_steps+1, B, 2)
        aux: auxiliary info dict
    """
    q0 = jnp.atleast_2d(jnp.asarray(q0, dtype=jnp.float64))
    p0 = jnp.atleast_2d(jnp.asarray(p0, dtype=jnp.float64))
    dt = jnp.asarray(dt, dtype=jnp.float64)

    mu, a, e, e_hat, q_hat, M0, n = _kepler_elements_from_state_2d(q0, p0, params)
    t = jnp.arange(num_steps + 1, dtype=jnp.float64) * dt

    def state_at_time(ti):
        M = M0 + n * ti
        M = jnp.arctan2(jnp.sin(M), jnp.cos(M))  # wrap to [-pi, pi]
        E = _solve_kepler_E(M, e)

        cE, sE = jnp.cos(E), jnp.sin(E)
        sqrt1me2 = jnp.sqrt(1.0 - e**2)

        r = a * (1.0 - e * cE)
        x = a * (cE - e)
        y = a * sqrt1me2 * sE

        fac = jnp.sqrt(mu * a) / r
        vx = -fac * sE
        vy = fac * sqrt1me2 * cE

        q = x[..., None] * e_hat + y[..., None] * q_hat
        v = vx[..., None] * e_hat + vy[..., None] * q_hat
        p = jnp.asarray(params.m, dtype=jnp.float64) * v
        return q, p

    q_traj, p_traj = jax.vmap(state_at_time)(t)  # (T+1, B, 2)
    aux = {"energy": kepler_hamiltonian(q_traj, p_traj, params)}
    return t, q_traj, p_traj, aux


def create_dataset(x0: jnp.ndarray, steps: jnp.ndarray, dt: float, params: KeplerParams):
    """Create dataset for training/testing.

    Args:
        x0: initial conditions (B, 2d) as (q0 | p0)
        steps: integer time indices at which to sample the flow (B,)
        dt: time step size
        params: Kepler problem parameters

    Returns:
        x0: initial conditions (B, 2d)
        x0_dot: initial time derivatives (B, 2d)
        xt: states at selected times (B, 2d)
        t: selected times (B,)
    """
    x0 = jnp.asarray(x0)
    steps = jnp.asarray(steps).astype(int)

    B, dim = x0.shape
    d = dim // 2
    q0 = x0[:, :d]
    p0 = x0[:, d:]

    max_step = int(steps.max(initial=0))
    t_full, q_full, p_full, _ = simulate_kepler_analytic(
        params=params,
        num_steps=max_step,
        dt=dt,
        q0=q0,
        p0=p0,
    )

    # Select states at specified time steps
    batch_idx = jnp.arange(B)
    q_sel = q_full[steps, batch_idx, :]
    p_sel = p_full[steps, batch_idx, :]
    xt = jnp.concatenate([q_sel, p_sel], axis=1)  # (B, 2d)
    t_sel = t_full[steps]

    # Compute initial time derivatives
    dqdt0, dpdt0 = kepler_equations(q0, p0, params)
    x0_dot = jnp.concatenate([dqdt0, dpdt0], axis=1)

    return x0, x0_dot, xt, t_sel


# ------------------------- Data builders -------------------------

@dataclass
class TrainConfig:
    n: int = 2  # spatial dimension (2D Kepler)
    dt: float = 0.01
    num_train: int = 5000
    num_test: int = 5000
    delta_t_max: float = 0.5
    train_t_dist: str = "LNP"  # uniform | step_scheduling | LNP
    mass_seed: int = 42
    model_seed: int = 0
    train_seed: int = 201
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 200
    log_every: int = 50
    w_q: float = 1.0
    w_p: float = 1.0
    w_action: float = 0.0
    w_hnn: float = 0.0
    action_loss_type: str = 'var'  # 'var' or 'quadratic_variation'


def sample_time_jumps(dist: str, dt: float, delta_t_max: float, key: jax.Array, size: int) -> jnp.ndarray:
    """Sample time jump steps according to specified distribution."""
    if dist == "uniform":
        max_jump = max(1, int(delta_t_max / dt))
        jumps = jax.random.randint(key, shape=(size,), minval=1, maxval=max_jump + 1)
    elif dist == "LNP":
        # Log-Normal Poisson distribution
        sigma2 = 0.5
        k1, k2 = jax.random.split(key)
        tau = jnp.sqrt(sigma2) * jax.random.normal(k1, shape=(size,)) + jnp.log(max(float(delta_t_max // dt - 1), 1.0)) - 0.5 * sigma2
        poisson = jax.random.poisson(k2, lam=jnp.exp(tau))
        jumps = poisson + 1
    else:  # step_scheduling
        unif = jax.random.uniform(key, shape=(size,), minval=0.0, maxval=1.0)
        ramp = (delta_t_max * jnp.arange(size) / size) // dt
        jumps = jnp.ceil(unif * ramp).astype(jnp.int32)
        jumps = jnp.maximum(jumps, 1)
    return jumps.astype(jnp.int32)


def build_datasets(cfg: TrainConfig, params: KeplerParams, test_steps: jnp.ndarray):
    """Build training and test datasets.

    Args:
        cfg: training configuration
        params: Kepler problem parameters
        test_steps: array of test time steps

    Returns:
        params: Kepler parameters
        train_dataset: (x0, x0_dot, xt, t) for training
        test_dataset: (x0, x0_dot, xt, t) for testing
        step_counts_test: test step counts for evaluation
    """
    key = jax.random.PRNGKey(cfg.mass_seed)
    q_key, p_key, dt_key, key = jax.random.split(key, 4)

    # Training data
    q0_train = jax.random.uniform(q_key, shape=(cfg.num_train, cfg.n), minval=-1.0, maxval=1.0)
    p0_train = jax.random.uniform(p_key, shape=(cfg.num_train, cfg.n), minval=-1.0, maxval=1.0)
    step_counts_train = sample_time_jumps(cfg.train_t_dist, cfg.dt, cfg.delta_t_max, dt_key, cfg.num_train)
    step_counts_train = jnp.maximum(step_counts_train, 1)
    train_dataset = create_dataset(
        x0=jnp.concatenate([q0_train, p0_train], axis=1),
        steps=step_counts_train,
        dt=cfg.dt,
        params=params,
    )

    # Test data
    q_test_key, p_test_key, _ = jax.random.split(key, 3)
    q0_test = jax.random.uniform(q_test_key, shape=(cfg.num_test, cfg.n), minval=-1.0, maxval=1.0)
    p0_test = jax.random.uniform(p_test_key, shape=(cfg.num_test, cfg.n), minval=-1.0, maxval=1.0)
    step_counts_test = jnp.tile(test_steps, cfg.num_test)
    test_dataset = create_dataset(
        x0=jnp.concatenate(
            [jnp.repeat(q0_test, repeats=len(test_steps), axis=0),
             jnp.repeat(p0_test, repeats=len(test_steps), axis=0)],
            axis=1,
        ),
        steps=step_counts_test,
        dt=cfg.dt,
        params=params,
    )
    return params, train_dataset, test_dataset, step_counts_test


# ------------------------- Training / Eval -------------------------

def build_train_step(model: MyActionAngleNetwork):
    """Build training step function."""
    def train_step(params, opt_state, dtst_batch, apply_fn, tx, loss_weights, action_loss_type='var'):
        x, x_dot, y, delta_t = dtst_batch
        n = x.shape[1] // 2
        q, p = x[:, :n], x[:, n:]
        q_dot, p_dot = x_dot[:, :n], x_dot[:, n:]

        def loss_fn(pp):
            q_, p_, I, _ = apply_fn({'params': pp}, q, p, delta_t, train=True)
            loss_q_se = ((1.0 / (1.0 + delta_t)) * ((q_ - y[:, :n]) ** 2).sum(axis=1)).mean()
            loss_p_se = ((1.0 / (1.0 + delta_t)) * ((p_ - y[:, n:]) ** 2).sum(axis=1)).mean()

            def H_hat_single(qi, pi):
                return model.apply({'params': pp}, qi[None, :], pi[None, :], method=model.hamiltonian).squeeze()

            def H_of_I_sum(I_in):
                return model.apply({'params': pp}, I_in, method=model.h_of_I).sum()

            omega = jax.grad(H_of_I_sum)(I)
            omega_norm = jnp.mean(jnp.linalg.norm(omega, axis=1))
            dHdq_model = jax.vmap(lambda qi, pi: jax.grad(H_hat_single, argnums=0)(qi, pi))(q, p)
            dHdp_model = jax.vmap(lambda qi, pi: jax.grad(H_hat_single, argnums=1)(qi, pi))(q, p)
            L_HNN = ((dHdq_model + p_dot) ** 2).mean() + ((dHdp_model - q_dot) ** 2).mean()

            if action_loss_type == 'quadratic_variation':
                L_action = jnp.sum(jnp.square(I[1:, :] - I[:-1, :])) / I.shape[0]
            else:
                L_action = lax.cond(
                    loss_weights[2] > 0.0,
                    lambda arr: jnp.var(arr, axis=0).sum(),
                    lambda arr: 0.0,
                    I,
                )

            total = (
                loss_weights[0] * loss_q_se
                + loss_weights[1] * loss_p_se
                + loss_weights[2] * L_action
                + loss_weights[3] * L_HNN
            )
            return total, omega_norm

        (loss, omega_norm), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return params, opt_state, loss, grad_norm, omega_norm

    return jax.jit(train_step, static_argnames=('apply_fn', 'tx'))


@jax.jit
def eval_batch(params, x, y, delta_t, apply_fn):
    """Evaluate model on a batch."""
    n = x.shape[1] // 2
    q, p = x[:, :n], x[:, n:]
    q_, p_, _, _ = apply_fn({'params': params}, q, p, delta_t, train=False)
    loss_q_se = ((1.0 / (1.0 + delta_t)) * ((q_ - y[:, :n]) ** 2).sum(axis=1)).sum()
    loss_p_se = ((1.0 / (1.0 + delta_t)) * ((p_ - y[:, n:]) ** 2).sum(axis=1)).sum()
    return loss_q_se + loss_p_se


def train_model(cfg: TrainConfig, model: MyActionAngleNetwork, train_dataset, test_dataset):
    """Train the model."""
    train_step_jit = build_train_step(model)
    variables = model.init(jax.random.PRNGKey(cfg.model_seed), jnp.zeros((1, cfg.n)), jnp.zeros((1, cfg.n)), jnp.asarray(cfg.dt))
    params = variables['params']
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=cfg.lr))
    opt_state = tx.init(params)

    x0_train, x0_dot_train, xt_train, t_train = train_dataset
    num_batches = x0_train.shape[0] // cfg.batch_size
    key = jax.random.PRNGKey(cfg.train_seed)

    for ep in range(cfg.epochs):
        perm_key, key = jax.random.split(jax.random.fold_in(key, ep))
        perm = jax.random.permutation(perm_key, x0_train.shape[0])
        x0_train = x0_train[perm]
        x0_dot_train = x0_dot_train[perm]
        xt_train = xt_train[perm]
        t_train = t_train[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            sl = slice(i * cfg.batch_size, (i + 1) * cfg.batch_size)
            batch = (x0_train[sl], x0_dot_train[sl], xt_train[sl], t_train[sl])
            params, opt_state, batch_loss, _, _ = train_step_jit(
                params,
                opt_state,
                batch,
                apply_fn=model.apply,
                tx=tx,
                loss_weights=(cfg.w_q, cfg.w_p, cfg.w_action, cfg.w_hnn),
                action_loss_type=cfg.action_loss_type,
            )
            epoch_loss += float(batch_loss)

            global_step = ep * num_batches + i + 1
            if global_step % cfg.log_every == 0 or (ep == cfg.epochs - 1 and i == num_batches - 1):
                # quick eval
                num_test_batches = test_dataset[0].shape[0] // cfg.batch_size
                test_losses = []
                for j in range(num_test_batches):
                    slt = slice(j * cfg.batch_size, (j + 1) * cfg.batch_size)
                    test_losses.append(float(eval_batch(
                        params,
                        test_dataset[0][slt],
                        test_dataset[2][slt],
                        test_dataset[3][slt],
                        apply_fn=model.apply,
                    )))
                if test_losses:
                    print(f"[step {global_step:05d}] eval_loss={sum(test_losses)/len(test_losses):.6f}")

        epoch_loss /= max(1, x0_train.shape[0])
        print(f"Epoch {ep + 1}/{cfg.epochs} - avg train loss: {epoch_loss:.6f}")

    return params


def evaluate_rmse_by_step(params, model, test_dataset, step_counts_test):
    """Evaluate RMSE by time step."""
    x_test, _, y_test, delta_t_test = test_dataset
    n_conf = x_test.shape[1] // 2
    q_test = x_test[:, :n_conf]
    p_test = x_test[:, n_conf:]
    q_target = y_test[:, :n_conf]
    p_target = y_test[:, n_conf:]
    q_pred, p_pred, _, _ = model.apply({'params': params}, q_test, p_test, delta_t_test, train=False)
    per_sample_q_rmse = jnp.sqrt(jnp.mean((q_pred - q_target) ** 2, axis=1))
    per_sample_p_rmse = jnp.sqrt(jnp.mean((p_pred - p_target) ** 2, axis=1))
    step_counts = np.asarray(step_counts_test)
    unique_steps = np.unique(step_counts)
    res = []
    for step in unique_steps:
        mask = step_counts == step
        res.append((int(step), float(per_sample_q_rmse[mask].mean()), float(per_sample_p_rmse[mask].mean())))
    return res


# ------------------------- Diagnostics -------------------------

def evaluate_hamiltonian_conservation(params, model, kepler_params: KeplerParams, cfg: TrainConfig, num_eval_samples: int = 100):
    """Evaluate Hamiltonian conservation on test trajectories.

    Args:
        params: trained model parameters
        model: MyActionAngleNetwork instance
        kepler_params: Kepler problem parameters
        cfg: training configuration
        num_eval_samples: number of trajectories to evaluate

    Returns:
        Dictionary with Hamiltonian statistics
    """
    key = jax.random.PRNGKey(999)
    q_key, p_key = jax.random.split(key)

    # Generate random initial conditions
    q0 = jax.random.uniform(q_key, shape=(num_eval_samples, cfg.n), minval=-1.0, maxval=1.0)
    p0 = jax.random.uniform(p_key, shape=(num_eval_samples, cfg.n), minval=-1.0, maxval=1.0)

    # Compute true Hamiltonian
    H_true = kepler_hamiltonian(q0, p0, kepler_params)

    # Compute predicted Hamiltonian
    H_pred = model.apply({'params': params}, q0, p0, method=model.hamiltonian)

    # Statistics
    H_error = jnp.abs(H_pred - H_true)
    results = {
        'H_true_mean': float(jnp.mean(H_true)),
        'H_true_std': float(jnp.std(H_true)),
        'H_pred_mean': float(jnp.mean(H_pred)),
        'H_pred_std': float(jnp.std(H_pred)),
        'H_error_mean': float(jnp.mean(H_error)),
        'H_error_std': float(jnp.std(H_error)),
        'H_relative_error': float(jnp.mean(H_error / jnp.abs(H_true))),
    }

    return results


def evaluate_action_angle(params, model, kepler_params: KeplerParams, cfg: TrainConfig, num_eval_samples: int = 100, num_steps: int = 100):
    """Evaluate action-angle variable conservation along trajectories.

    Args:
        params: trained model parameters
        model: MyActionAngleNetwork instance
        kepler_params: Kepler problem parameters
        cfg: training configuration
        num_eval_samples: number of trajectories to evaluate
        num_steps: number of time steps to evaluate along trajectory

    Returns:
        Dictionary with action-angle statistics
    """
    key = jax.random.PRNGKey(888)
    q_key, p_key = jax.random.split(key)

    # Generate random initial conditions
    q0 = jax.random.uniform(q_key, shape=(num_eval_samples, cfg.n), minval=-1.0, maxval=1.0)
    p0 = jax.random.uniform(p_key, shape=(num_eval_samples, cfg.n), minval=-1.0, maxval=1.0)

    # Simulate trajectories
    t, q_traj, p_traj, _ = simulate_kepler_analytic(
        params=kepler_params,
        num_steps=num_steps,
        dt=cfg.dt,
        q0=q0,
        p0=p0,
    )

    # Compute actions along trajectories
    # Flatten time and batch dimensions
    q_flat = q_traj.reshape(-1, cfg.n)
    p_flat = p_traj.reshape(-1, cfg.n)

    _, _, I_pred, _ = model.apply({'params': params}, q_flat, p_flat, jnp.asarray(0.0), train=False)
    I_pred = I_pred.reshape(num_steps + 1, num_eval_samples, cfg.n)

    # Compute variance of actions over time for each trajectory
    I_variance = jnp.var(I_pred, axis=0)  # (num_eval_samples, n)

    results = {
        'I_variance_mean': float(jnp.mean(I_variance)),
        'I_variance_std': float(jnp.std(I_variance)),
        'I_mean': float(jnp.mean(I_pred)),
        'I_std': float(jnp.std(I_pred)),
    }

    return results


# ------------------------- CLI / Main -------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Train MyActionAngleNetwork on Kepler problem (flexible)")

    # data
    p.add_argument('--n', type=int, default=2, help='Spatial dimension (2 for 2D Kepler)')
    p.add_argument('--dt', type=float, default=0.01, help='Base step size')
    p.add_argument('--num-train', type=int, default=5000)
    p.add_argument('--num-test', type=int, default=5000)
    p.add_argument('--delta-t-max', type=float, default=0.5, help='Max jump horizon in seconds')
    p.add_argument('--train-dist', type=str, default='LNP', choices=['step_scheduling', 'uniform', 'LNP'])
    p.add_argument('--test-steps', type=parse_int_list, default=[1, 2, 5, 10, 20, 50])

    # physical params (Kepler)
    p.add_argument('--mass', type=float, default=1.0, help='Mass of orbiting body')
    p.add_argument('--k', type=float, default=1.0, help='Gravitational parameter')
    p.add_argument('--eps', type=float, default=1e-8, help='Softening parameter')

    # model hyperparams
    p.add_argument('--dim-hidden', type=int, default=128)
    p.add_argument('--dim-hidden-list', type=parse_int_list, default=[32, 128, 32])
    p.add_argument('--num-gsblocks', type=int, default=30)
    p.add_argument('--type-polar', type=str, default='canonical', choices=['canonical', 'normal'])
    p.add_argument('--activation', type=str, default='silu', choices=['relu', 'tanh', 'sigmoid', 'silu'])
    p.add_argument('--theta-predictor', type=str, default='gradient', choices=['gradient', 'mlp'])
    p.add_argument('--mlp-res-connection', type=str2bool, default=False)
    p.add_argument('--learn-scale', type=str2bool, default=True)

    # train loop
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--log-every', type=int, default=50)

    # loss weights
    p.add_argument('--w-q', type=float, default=1.0)
    p.add_argument('--w-p', type=float, default=1.0)
    p.add_argument('--w-action', type=float, default=0.0)
    p.add_argument('--w-hnn', type=float, default=0.0)
    p.add_argument('--action-loss-type', type=str, default='var', choices=['var', 'quadratic_variation'])

    # seeds
    p.add_argument('--mass-seed', type=int, default=42)
    p.add_argument('--model-seed', type=int, default=0)
    p.add_argument('--train-seed', type=int, default=201)

    # diagnostics
    p.add_argument('--run-diagnostics', type=str2bool, default=True, help='Run Hamiltonian and action-angle diagnostics')
    p.add_argument('--num-diag-samples', type=int, default=100, help='Number of samples for diagnostics')
    p.add_argument('--num-diag-steps', type=int, default=100, help='Number of steps for action-angle diagnostics')

    return p


def main():
    args = build_argparser().parse_args()

    # Enable 64-bit precision for better numerical accuracy
    jax.config.update("jax_enable_x64", True)

    # configure
    cfg = TrainConfig(
        n=args.n,
        dt=args.dt,
        num_train=args.num_train,
        num_test=args.num_test,
        delta_t_max=args.delta_t_max,
        train_t_dist=args.train_dist,
        mass_seed=args.mass_seed,
        model_seed=args.model_seed,
        train_seed=args.train_seed,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_every=args.log_every,
        w_q=args.w_q,
        w_p=args.w_p,
        w_action=args.w_action,
        w_hnn=args.w_hnn,
        action_loss_type=args.action_loss_type,
    )

    # physical params
    kepler_params = set_kepler_params(m=args.mass, k=args.k, eps=args.eps)
    test_steps = jnp.asarray(args.test_steps, dtype=jnp.int32)

    # model
    def get_act(name: str):
        name = name.lower()
        if name == 'relu':
            return jax.nn.relu
        if name == 'tanh':
            return jax.nn.tanh
        if name == 'sigmoid':
            return jax.nn.sigmoid
        return jax.nn.silu

    model = MyActionAngleNetwork(
        dim_config=cfg.n,
        dim_hidden=args.dim_hidden,
        dim_hidden_list=args.dim_hidden_list,
        num_gsblocks=args.num_gsblocks,
        type_polar=args.type_polar,
        activation=get_act(args.activation),
        mlp_res_connection=args.mlp_res_connection,
        theta_predictor=args.theta_predictor,
        learn_scale=args.learn_scale,
    )

    # data
    print("Building datasets...")
    kepler_params, train_dataset, test_dataset, step_counts_test = build_datasets(cfg, kepler_params, test_steps)

    # train
    print("\nTraining model...")
    learned_params = train_model(cfg, model, train_dataset, test_dataset)

    # eval
    print("\nEvaluating RMSE by prediction horizon...")
    rmse_table = evaluate_rmse_by_step(learned_params, model, test_dataset, step_counts_test)
    print("\nRMSE by prediction horizon (steps):")
    for step, q_rmse, p_rmse in rmse_table:
        print(f"  {step:2d}-step -> q RMSE={q_rmse:.4e}, p RMSE={p_rmse:.4e}")

    # diagnostics
    if args.run_diagnostics:
        print("\n" + "=" * 60)
        print("Running diagnostics...")
        print("=" * 60)

        print("\n1. Hamiltonian Conservation:")
        h_results = evaluate_hamiltonian_conservation(
            learned_params, model, kepler_params, cfg,
            num_eval_samples=args.num_diag_samples
        )
        for key, val in h_results.items():
            print(f"  {key}: {val:.6e}")

        print("\n2. Action-Angle Variables:")
        aa_results = evaluate_action_angle(
            learned_params, model, kepler_params, cfg,
            num_eval_samples=args.num_diag_samples,
            num_steps=args.num_diag_steps
        )
        for key, val in aa_results.items():
            print(f"  {key}: {val:.6e}")

        print("\n" + "=" * 60)
        print("Diagnostics complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
