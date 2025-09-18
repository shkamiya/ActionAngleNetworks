import os, shutil, pathlib, json
import argparse
import numpy as np
import jax
from jax import value_and_grad, jit, vmap, grad, random
import jax.numpy as jnp
#from flax.training import train_state
from flax import serialization
import flax.linen as nn
import matplotlib.pyplot as plt
import optax

from typing import Any, Callable, Sequence, Optional, List, Tuple
from pathlib import Path

import time
import datetime
import wandb

# from action_angle_networks import train, analysis
# from action_angle_networks.simulation import harmonic_motion_simulation as hsim
# from action_angle_networks.configs.harmonic_motion import default as hm_default

from action_angle_networks.sk_layers import (
    GSBlock, GSympNet, MLP, MLPFlexible,
    PolarCoordinates,  InversePolarCoordinates,
    GotosCanonicalPolarCoordinates, InverseGotosCanonicalPolarCoordinates,
)
from action_angle_networks.sk_models import MyActionAngleNetwork

# %config InlineBackend.figure_format = 'retina'
# logging.set_verbosity(logging.INFO)
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

## Helper functions
def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ["0", "n", "no", "f", "false"]:
        return False
    elif x[0] in ["1", "y", "yes", "t", "true"]:
        return True
    raise ValueError("Invalid value: {}".format(x))

def _atomic_save(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(serialization.to_bytes(params))
    os.replace(tmp, path)  # atomic

## Helper classes and functions for simulating coupled 1D harmonic oscillators
@dataclass
class HarmonicParams:
    n: int
    m: np.ndarray         # (n,) masses per oscillator
    k_wall: float         # wall spring constant
    k_pair: float         # pairwise coupling spring constant
    A_modes: np.ndarray   # (n,) amplitudes per normal mode
    phi_modes: np.ndarray # (n,) phases per normal mode (radians)

def sample_harmonic_params(
    n: int,
    m_range: Tuple[float, float] = (1.0, 5.0),
    k_wall: float = 0.01,
    k_pair: float = 0.005,
    A_range: Tuple[float, float] = (1.0, 3.0),
    seed: Optional[int] = 0,
) -> HarmonicParams:
    rng = np.random.default_rng(seed)
    m = rng.uniform(*m_range, size=(n,)).astype(np.float64)
    # Normal-mode amplitudes/phases
    A_modes = rng.uniform(*A_range, size=(n,)).astype(np.float64)
    phi_modes = rng.uniform(0.0, 2*np.pi, size=(n,)).astype(np.float64)
    return HarmonicParams(
        n=n, m=m, k_wall=float(k_wall), k_pair=float(k_pair),
        A_modes=A_modes, phi_modes=phi_modes
    )

def _normal_mode_eigendecomposition(params: HarmonicParams):
    """
    Returns eigenvalues/eigenvectors for q¨ = M^{-1} L q.
    Repoに合わせて L = -(k_wall + n*k_pair) I + k_pair 1 1^T を用い、
    固有値は負になる（w = sqrt(-eigvals)）。
    """
    n = params.n
    I = np.eye(n)
    J = np.ones((n, n))
    L = -(params.k_wall + n * params.k_pair) * I + params.k_pair * J
    M_inv = np.diag(1.0 / params.m)
    M_inv_L = M_inv @ L
    eigvals, eigvecs = np.linalg.eig(M_inv_L)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    return eigvals, eigvecs  # (n,), (n,n)

def simulate_coupled_1d_harmonic(
    params: HarmonicParams, num_steps: int, dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      t: (T,)
      q: (T, n) positions
      p: (T, n) momenta  (p = m * dq/dt)
      aux: dict with eigvals/eigvecs/w
    """
    n = params.n
    t = np.arange(num_steps, dtype=np.float64) * float(dt)

    # Normal modes
    eigvals, V = _normal_mode_eigendecomposition(params)
    w = np.sqrt(np.clip(-eigvals, a_min=0.0, a_max=None))  # angular freq

    # Mode-space trajectories
    Wt_phi = t[:, None] * w[None, :] + params.phi_modes[None, :]  # (T, n)
    q_mode = params.A_modes[None, :] * np.cos(Wt_phi)              # (T, n)
    qdot_mode = -params.A_modes[None, :] * (w[None, :]) * np.sin(Wt_phi)  # (T, n)

    # Map back to physical coords: q = V q_mode, qdot = V qdot_mode
    q = q_mode @ V.T
    qdot = qdot_mode @ V.T
    p = qdot * params.m[None, :]  # momentum = m * qdot

    aux = {"eigvals": eigvals, "eigvecs": V, "w": w}
    return t, q, p, aux



@jit
def train_step(model, params, opt_state, x, y, delta_t, alpha, key):
    # x: (B, 2n) current (q, p)
    # y: (B, 2n) target  (q, p)
    T = x.shape[0]
    n = x.shape[1] // 2
    q, p = x[:, :n], x[:, n:]  # (B, n), (B, n)

    def loss_fn(pp):
        q_, p_, I, _ = model.apply({'params': pp}, q, p, delta_t, train=True)#, rngs={'noise': key})
        #x_ = jnp.concatenate([q_, p_], axis=-1)
        # q_, p_: (B, n), (B, n)
        # I: (B, n) actions
        
        # prediction loss
        loss_q_se = (1./(1.+delta_t))*((q_ - y[:, :n]) ** 2).mean()#.sum()#
        loss_p_se = (1./(1.+delta_t))*((p_ - y[:, n:]) ** 2).mean()#.sum()#

        # actions should be constant
        loss_action = jnp.var(I, axis=0).sum()

        return loss_q_se + loss_p_se + alpha * loss_action

    loss, grads = value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jit
def eval_batch(model, params, x, y, delta_t, key):
    n = x.shape[1] // 2
    q, p = x[:, :n], x[:, n:]  # (B, 3), (B, 3)
    
    q_, p_, _, _ = model.apply({'params': params}, q, p, delta_t, train=False)
    #x_ = jnp.concatenate([q_, p_], axis=-1)

    loss_q_mse = ((q_ - y[:, :n]) ** 2).mean()
    loss_p_mse = ((p_ - y[:, n:]) ** 2).mean()

    return loss_q_mse + loss_p_mse

def main():
    parser = argparse.ArgumentParser(description='Train Action-Angle Network on Harmonic Motion Data')

    # Exp name & Wandb
    parser.add_argument('--experiment-name', type=str, default='aan_harmonic', help='Experiment name')
    parser.add_argument('--wandb-project', default='aan_harmonic', help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, help='W&B entity name')
    parser.add_argument('--no-wandb', type=str2bool, default=False, help='Disable W&B logging')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name (default: use exp_name and job ID)')

    # Simulating Data
    parser.add_argument('--n', type=int, default=2, help='Number of oscillators')
    parser.add_argument('--T', type=int, default=1000, help='Number of samples')
    parser.add_argument('--delta-t', type=float, default=1.0, help='Time delta between samples')
    
    parser.add_argument('--k-wall', type=float, default=0.01, help='Wall spring constant')
    parser.add_argument('--k-pair', type=float, default=0.005, help='Pairwise spring constant')
    parser.add_argument('--m-min', type=float, default=1.0, help='Minimum mass')
    parser.add_argument('--m-max', type=float, default=3.0, help='Maximum mass')
    parser.add_argument('--A-min', type=float, default=1.0, help='Minimum amplitude')
    parser.add_argument('--A-max', type=float, default=2.0, help='Maximum amplitude')

    # Model
    parser.add_argument('--dim-hidden', type=int, default=100, help='Hidden dimension size')
    parser.add_argument('--num-gsblocks', type=int, default=20, help='Number of GSBlocks in the model')
    parser.add_argument('--type-polar', type=str, default='canonical', choices=['normal', 'canonical'], help='Type of polar coordinates') 
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['relu', 'tanh', 'sigmoid'], help='Activation function')
    parser.add_argument('--theta-predictor', type=str, default='gradient', choices=['gradient', 'mlp'], help='Theta predictor type')
    parser.add_argument('--mlp-res-connection', type=str2bool, default=False, help='Use residual connections in MLPs')
    parser.add_argument('--dim-hidden-list', type=int, nargs='+', default=[64, 64, 64], help='List of hidden dimensions for MLPs')


    # Train
    parser.add_argument('--train-split', type=float, default=0.1, help='Proportion of data for training')
    parser.add_argument('--test-split', type=float, default=0.5, help='Proportion of data for testing')    
    parser.add_argument('--single-step', type=str2bool, default=True, help='Use single step predictions')
    parser.add_argument('--num-steps', type=int, default=5000, help='Number of training steps')
    parser.add_argument('--delta-t-max', type=float, default=10.0, help='Maximum delta_t for training')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--test-time-jumps', type=int, nargs='+', default=[1, 2, 5, 10, 20, 50], help='List of delta_t for testing')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization weight for action variance')
    parser.add_argument('--normalize-qp', type=str2bool, default=False, help='Normalize (q, p) before feeding into GSympNet ')

    # Logging and saving
    parser.add_argument('--log-every', type=int, default=100, help='Evaluation cadence (in steps)')
    parser.add_argument('--save-every', type=int, default=1000, help='Model saving cadence (in steps)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results (default: use W&B run ID)')

    # seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for simulation')


    args = parser.parse_args()

    # Run simulation
    params = sample_harmonic_params(
        n=args.n,
        m_range=(args.m_min, args.m_max),
        k_wall=args.k_wall,
        k_pair=args.k_pair,
        A_range=(args.A_min, args.A_max),
        seed=args.seed
    )
    t, q, p, aux = simulate_coupled_1d_harmonic(params, num_steps=args.T, dt=args.delta_t)
    
    print(f"Training on {args.n}-dim harmonic oscillator data with {args.T} samples, dt={args.delta_t}")
    print(f"Masses: {params.m}")
    print(f"q shape: {q.shape}, p shape: {p.shape}")

    # Split data
    T_train = int(args.T * args.train_split)
    T_test  = int(args.T * args.test_split)

    q_train = jnp.asarray(q[:T_train])
    p_train = jnp.asarray(p[:T_train])

    q_test  = jnp.asarray(q[-T_test:])
    p_test  = jnp.asarray(p[-T_test:])


    
    model = MyActionAngleNetwork(
        dim_config=args.n,
        dim_hidden=args.dim_hidden,
        num_gsblocks=args.num_gsblocks,
        type_polar=args.type_polar,
        activation=getattr(nn, args.activation),
        mlp_res_connection=args.mlp_res_connection,
        theta_predictor=args.theta_predictor,
        dim_hidden_list=args.dim_hidden_list,
    )

    print(
        f"AAN Model created with {args.num_gsblocks} GSBlocks, hidden dim {args.dim_hidden}, polar type {args.type_polar}"
        f", theta predictor {args.theta_predictor}, mlp_res_connection {args.mlp_res_connection}")


    key = jax.random.PRNGKey(args.seed)
    eval_key = jax.random.PRNGKey(args.seed + 123)

    q0 = jnp.zeros((1, int(args.n)))
    p0 = jnp.zeros((1, int(args.n)))
    delta_t0 = jnp.asarray(args.delta_t)

    variables = model.init(key, q0, p0, delta_t0)
    params = variables['params']
    num_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, params)))
    print(f'Total parameters: {num_params}')


    # wandb
    if not args.no_wandb:
        wandb_config = {
            'n': args.n,
            'T': args.T,
            'delta_t': args.delta_t,
            'k_wall': args.k_wall,
            'k_pair': args.k_pair,
            'm_min': args.m_min,
            'm_max': args.m_max,
            'A_min': args.A_min,
            'A_max': args.A_max,
            'dim_hidden': args.dim_hidden,
            'num_gsblocks': args.num_gsblocks,
            'type_polar': args.type_polar,
            'activation': args.activation,
            'theta_predictor': args.theta_predictor,
            'mlp_res_connection': args.mlp_res_connection,
            'dim_hidden_list': args.dim_hidden_list,
            'train_split': args.train_split,
            'test_split': args.test_split,
            'single_step': args.single_step,
            'num_steps': args.num_steps,
            'delta_t_max': args.delta_t_max,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'test_time_jumps': args.test_time_jumps,
            'alpha': args.alpha,
            'normalize_qp': args.normalize_qp,
            'log_every': args.log_every,
            'save_every': args.save_every,
            'save_dir': args.save_dir,
            'experiment_name': args.experiment_name,
            'num_params': num_params,
            'seed': args.seed
        }



        job_id = os.environ.get("PBS_JOBID") or os.environ.get("PJM_JOBID") or "local"
        if args.wandb_run_name is not None:
            wandb_run_name = args.wandb_run_name.format(**vars(args), job_id=job_id)
        else:
            wandb_run_name = f"{wandb_config['exp_name']}_job{job_id}"
        #run_name = args.wandb_run_name if args.wandb_run_name else f"{args.experiment_name}_T{args.T}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=wandb_run_name,
        )
    
    run = wandb.run if not args.no_wandb else None

    # save_dirを決める（Run ID 基準）
    if not args.no_wandb:
        if args.save_dir is None:
            args.save_dir = f"./results/{args.experiment_name}/{run.id}"
        # W&B上のconfigにも反映（後から見返せるように）
        wandb.config.update({"save_dir": args.save_dir}, allow_val_change=True)
    else:
        if args.save_dir is None:
            current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            args.save_dir = f"./results/{args.experiment_name}_{current_time}"
    jobdir = args.save_dir

    os.makedirs(jobdir, exist_ok=True)


    # オプティマイザ
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)


    print("Starting training...")
    print(f"num_steps: {args.num_steps}, delta_t_max: {args.delta_t_max}, batch_size: {args.batch_size}, log_every: {args.log_every}")
    print(f"test_time_jumps: {args.test_time_jumps}")
    print(f"alpha (weight for action constancy loss): {args.alpha}")



    for step in range(1, args.num_steps + 1):
        key, sub1, sub2 = jax.random.split(key, 3)

        # jump幅をサンプル
        max_time_jump_for_step = 1 + (step / args.num_steps) * (args.delta_t_max - 1)
        jump = jax.random.randint(sub1, shape=(), minval=1, maxval=int(max_time_jump_for_step) + 1)
        delta_t_scalar = jnp.asarray(args.delta_t * jump)

        # n_train - jump このペアを作る
        curr_q, curr_p = q_train[:-jump], p_train[:-jump]
        tgt_q, tgt_p = q_train[jump:], p_train[jump:]

        # 重複を許さないでbatch_size個くらいサンプル
        batch_size = min(args.batch_size, curr_q.shape[0]) # curr_qの数がとても小さいときはそれを上限にする
        idx_sample = jax.random.choice(
            sub2, curr_q.shape[0], shape=(batch_size,), replace=False
        )
        batch = (
            curr_q[idx_sample], curr_p[idx_sample],
            tgt_q[idx_sample], tgt_p[idx_sample],
        )

        #batch = sample_train_batch(sub)
        # batch = train_curr_q[idx], train_curr_p[idx], train_tgt_q[idx],  train_tgt_p[idx],

        params, opt_state, train_loss = train_step(
            model=model,
            params=params,
            opt_state=opt_state,
            x=jnp.concatenate([batch[0], batch[1]], axis=-1),
            y=jnp.concatenate([batch[2], batch[3]], axis=-1),
            alpha=args.alpha,
            delta_t=delta_t_scalar,
            key=key
        )

        if not args.no_wandb:
            wandb.log({
                'train/loss': float(train_loss),
                'train/step': step,
                'train/delta_t': float(delta_t_scalar),
            }, step=step)

        if step % args.log_every == 0 or step == 1 or step == args.num_steps:

            test_loss_list = []
            for jump in args.test_time_jumps:
                test_curr_q, test_curr_p = q_test[:-jump], p_test[:-jump] # (T-jump, n)
                test_tgt_q, test_tgt_p = q_test[jump:], p_test[jump:] # (T-jump, n)
                delta_t_test = args.delta_t * jump
                test_loss_jump = eval_batch(
                    model=model,
                    params=params,
                    x=jnp.concatenate([test_curr_q, test_curr_p], axis=-1), 
                    y=jnp.concatenate([test_tgt_q, test_tgt_p], axis=-1),
                    delta_t=delta_t_test,
                    key=eval_key
                )
                test_loss_list.append(test_loss_jump)
            
            test_str = ", ".join(f"jump={j}: {float(l):.6f}"
                                for j, l in zip(args.test_time_jumps, test_loss_list))
            print(f"step {step:5d}  train_loss {float(train_loss):.6f}  test_loss [{test_str}]")


            if not args.no_wandb:
                for j, l in zip(args.test_time_jumps, test_loss_list):
                    wandb.log({
                        f'test/loss_jump_{j}': float(l),
                        'train/step': step,
                    }, step=step)


        save_dir = Path(args.save_dir)
        if step % args.save_every == 0 or step == args.num_steps:
            # 定期的にモデルを保存
            _atomic_save(save_dir / 'checkpoints' / f'params_step_{step:05d}.params', params)
            print(f"[ckpt] saved: {save_dir / 'checkpoints' / f'params_step_{step:05d}.params'}")


    if not args.no_wandb:
        wandb.log({
            "final/train_loss": float(train_loss),
            "final/step": step,
        }, step=step)
        for j, l in zip(args.test_time_jumps, test_loss_list):
            wandb.log({
                f'final/test_loss_jump_{j}': float(l),
            }, step=step)

    print("Training completed.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()