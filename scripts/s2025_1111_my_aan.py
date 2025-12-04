#!/usr/bin/env python3
"""
Flexible trainer for MyActionAngleNetwork (AAN) on synthetic harmonic-oscillator
data, adapted from the first training setup in
notebooks/e2025_1109_hhn_vs_aan_cleaner.ipynb and argument style inspired by
scripts/s2025_0918_aan_harmonic_motion.py.

Key features
- Dataset from random (q0,p0) and variable time jumps (1..ceil(delta_t_max/dt)).
- Analytic ground-truth flow for coupled 1D harmonic oscillators.
- Configurable model hyperparameters and loss weights.
- Horizon-wise RMSE (q/p) evaluation at custom test steps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, List

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


# ------------------------- Physics helpers -------------------------

@dataclass
class HarmonicParams:
    n: int
    m: np.ndarray
    k_wall: float
    k_pair: float


def set_harmonic_params(n: int, m: Sequence[float], k_wall: float, k_pair: float) -> HarmonicParams:
    return HarmonicParams(n=n, m=np.asarray(m, dtype=np.float64), k_wall=float(k_wall), k_pair=float(k_pair))


def _normal_mode_eigendecomposition(params: HarmonicParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n = params.n
    eye = jnp.eye(n)
    ones = jnp.ones((n, n))
    L = -(params.k_wall + n * params.k_pair) * eye + params.k_pair * ones
    m_vec = jnp.asarray(params.m)
    m_sqrt_inv = jnp.diag(1.0 / jnp.sqrt(m_vec))
    sym = m_sqrt_inv @ L @ m_sqrt_inv
    eigvals, U = jnp.linalg.eigh(sym)
    V = m_sqrt_inv @ U
    return eigvals.real, V.real


def flow_coupled_1d_harmonic_init_values(
    params: HarmonicParams,
    num_steps: jnp.ndarray,
    dt: float,
    q0: jnp.ndarray,
    p0: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    t = num_steps.astype(jnp.float64) * float(dt)
    eigvals, V = _normal_mode_eigendecomposition(params)
    w = jnp.sqrt(jnp.clip(-eigvals, a_min=0.0, a_max=None))

    # amplitudes/phases per sample
    M = jnp.diag(jnp.asarray(params.m))
    q_mode0 = (q0 @ M) @ V
    qdot_mode0 = p0 @ V
    w_safe = jnp.where(w < 1e-12, 1e-12, w)
    A = jnp.sqrt(q_mode0**2 + (qdot_mode0 / w_safe) ** 2)
    phi = jnp.arctan2(-qdot_mode0 / w_safe, q_mode0)

    wt_phi = t[:, None] * w_safe[None, :] + phi
    q_mode = A * jnp.cos(wt_phi)
    qdot_mode = -A * w_safe[None, :] * jnp.sin(wt_phi)
    q = q_mode @ V.T
    qdot = qdot_mode @ V.T
    p = qdot * params.m[None, :]
    aux = {"eigvals": eigvals, "eigvecs": V, "w": w}
    return t, q, p, aux


def create_dataset(x0: jnp.ndarray, steps: jnp.ndarray, dt: float, params: HarmonicParams):
    B, dim = x0.shape
    n = dim // 2
    q0 = x0[:, :n]
    p0 = x0[:, n:]
    t, q, p, _ = flow_coupled_1d_harmonic_init_values(
        params=set_harmonic_params(n=n, m=params.m, k_wall=params.k_wall, k_pair=params.k_pair),
        num_steps=steps,
        dt=dt,
        q0=q0,
        p0=p0,
    )
    xt = jnp.concatenate([q, p], axis=1)
    dqdt0 = p0 / params.m[None, :]
    eye = np.eye(n)
    ones = np.ones((n, n))
    L = -(params.k_wall + n * params.k_pair) * eye + params.k_pair * ones
    dpdt0 = jnp.matmul(q0, jnp.asarray(L).T)
    x0_dot = jnp.concatenate([dqdt0, dpdt0], axis=1)
    return x0, x0_dot, xt, t


# ------------------------- Data builders -------------------------

@dataclass
class TrainConfig:
    n: int = 2
    dt: float = 0.01
    num_train: int = 5000
    num_test: int = 5000
    delta_t_max: float = 0.1
    train_t_dist: str = "step_scheduling"  # uniform | step_scheduling | LNP
    mass_seed: int = 42
    model_seed: int = 0
    train_seed: int = 201
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    log_every: int = 50
    w_q: float = 1.0
    w_p: float = 1.0
    w_action: float = 0.0
    w_hnn: float = 0.0
    action_loss_type: str = 'var'  # 'var' or 'quadratic_variation'


def sample_time_jumps(dist: str, dt: float, delta_t_max: float, key: jax.Array, size: int) -> jnp.ndarray:
    if dist == "uniform":
        max_jump = max(1, int(delta_t_max / dt))
        jumps = jax.random.randint(key, shape=(size,), minval=1, maxval=max_jump + 1)
    elif dist == "LNP":
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


def build_datasets(cfg: TrainConfig, params: HarmonicParams, mass_range: Tuple[float, float] | None, masses_list: List[float] | None, test_steps: jnp.ndarray):
    key = jax.random.PRNGKey(cfg.mass_seed)
    q_key, p_key, dt_key, key = jax.random.split(key, 4)

    # masses handling
    if masses_list is not None and len(masses_list) == cfg.n:
        masses = np.asarray(masses_list, dtype=np.float64)
    else:
        lo, hi = mass_range if mass_range is not None else (0.1, 0.3)
        rng = np.random.default_rng(cfg.mass_seed)
        masses = rng.uniform(lo, hi, size=(cfg.n,)).astype(np.float64)
    params = set_harmonic_params(cfg.n, masses, params.k_wall, params.k_pair)

    q0_train = jax.random.uniform(q_key, shape=(cfg.num_train, cfg.n), minval=-3.0, maxval=3.0)
    p0_train = jax.random.uniform(p_key, shape=(cfg.num_train, cfg.n), minval=-3.0, maxval=3.0)
    step_counts_train = sample_time_jumps(cfg.train_t_dist, cfg.dt, cfg.delta_t_max, dt_key, cfg.num_train)
    step_counts_train = jnp.maximum(step_counts_train, 1)
    train_dataset = create_dataset(
        x0=jnp.concatenate([q0_train, p0_train], axis=1),
        steps=step_counts_train,
        dt=cfg.dt,
        params=params,
    )

    # test
    q_test_key, p_test_key, _ = jax.random.split(key, 3)
    q0_test = jax.random.uniform(q_test_key, shape=(cfg.num_test, cfg.n), minval=-3.0, maxval=3.0)
    p0_test = jax.random.uniform(p_test_key, shape=(cfg.num_test, cfg.n), minval=-3.0, maxval=3.0)
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
    def train_step(params, opt_state, dtst_batch, apply_fn, tx, loss_weights, action_loss_type='var'):
        x, x_dot, y, delta_t = dtst_batch
        n = x.shape[1] // 2
        q, p = x[:, :n], x[:, n:]
        q_dot, p_dot = x_dot[:, :n], x_dot[:, n:]

        def loss_fn(pp):
            q_, p_, I, _ = apply_fn({'params': pp}, q, p, delta_t, train=True)
            loss_q_se = ((1.0 / (1.0 + delta_t)) * ((q_ - y[:, :n]) ** 2).sum(axis=1)).sum()
            loss_p_se = ((1.0 / (1.0 + delta_t)) * ((p_ - y[:, n:]) ** 2).sum(axis=1)).sum()

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
    n = x.shape[1] // 2
    q, p = x[:, :n], x[:, n:]
    q_, p_, _, _ = apply_fn({'params': params}, q, p, delta_t, train=False)
    loss_q_se = ((1.0 / (1.0 + delta_t)) * ((q_ - y[:, :n]) ** 2).sum(axis=1)).sum()
    loss_p_se = ((1.0 / (1.0 + delta_t)) * ((p_ - y[:, n:]) ** 2).sum(axis=1)).sum()
    return loss_q_se + loss_p_se


def train_model(cfg: TrainConfig, model: MyActionAngleNetwork, train_dataset, test_dataset):
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


# ------------------------- CLI / Main -------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Train MyActionAngleNetwork on harmonic data (flexible)")

    # data
    p.add_argument('--n', type=int, default=2, help='DOF (oscillators)')
    p.add_argument('--dt', type=float, default=0.01, help='Base step size')
    p.add_argument('--num-train', type=int, default=5000)
    p.add_argument('--num-test', type=int, default=5000)
    p.add_argument('--delta-t-max', type=float, default=0.1, help='Max jump horizon in seconds')
    p.add_argument('--train-dist', type=str, default='step_scheduling', choices=['step_scheduling', 'uniform', 'LNP'])
    p.add_argument('--test-steps', type=parse_int_list, default=[1,2,5,10,20,50])

    # physical params
    p.add_argument('--k-wall', type=float, default=5.0)
    p.add_argument('--k-pair', type=float, default=2.5)
    p.add_argument('--masses', type=parse_float_list, default=None, help='Explicit masses list (length n)')
    p.add_argument('--m-min', type=float, default=0.1)
    p.add_argument('--m-max', type=float, default=0.3)

    # model hyperparams
    p.add_argument('--dim-hidden', type=int, default=64)
    p.add_argument('--dim-hidden-list', type=parse_int_list, default=[32,32,32])
    p.add_argument('--num-gsblocks', type=int, default=20)
    p.add_argument('--type-polar', type=str, default='canonical', choices=['canonical','normal'])
    p.add_argument('--activation', type=str, default='silu', choices=['relu','tanh','sigmoid','silu'])
    p.add_argument('--theta-predictor', type=str, default='gradient', choices=['gradient','mlp'])
    p.add_argument('--mlp-res-connection', type=str2bool, default=False)
    p.add_argument('--learn-scale', type=str2bool, default=False)

    # train loop
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--log-every', type=int, default=50)

    # loss weights
    p.add_argument('--w-q', type=float, default=1.0)
    p.add_argument('--w-p', type=float, default=1.0)
    p.add_argument('--w-action', type=float, default=0.0)
    p.add_argument('--w-hnn', type=float, default=0.0)
    p.add_argument('--action-loss-type', type=str, default='var', choices=['var','quadratic_variation'])

    # seeds
    p.add_argument('--mass-seed', type=int, default=42)
    p.add_argument('--model-seed', type=int, default=0)
    p.add_argument('--train-seed', type=int, default=201)
    return p


def main():
    args = build_argparser().parse_args()

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

    # physical params (masses handled in builder)
    params = set_harmonic_params(n=cfg.n, m=[1.0]*cfg.n, k_wall=args.k_wall, k_pair=args.k_pair)
    masses_list = args.masses if args.masses is not None else None
    mass_range = (args.m_min, args.m_max) if masses_list is None else None
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
    params, train_dataset, test_dataset, step_counts_test = build_datasets(cfg, params, mass_range, masses_list, test_steps)

    # train
    learned_params = train_model(cfg, model, train_dataset, test_dataset)

    # eval
    rmse_table = evaluate_rmse_by_step(learned_params, model, test_dataset, step_counts_test)
    print("\nRMSE by prediction horizon (steps):")
    for step, q_rmse, p_rmse in rmse_table:
        print(f"  {step:2d}-step -> q RMSE={q_rmse:.4e}, p RMSE={p_rmse:.4e}")


if __name__ == "__main__":
    main()

