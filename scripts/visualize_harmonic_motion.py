#!/usr/bin/env python3
# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Action-Angle Networks Visualization Script for Harmonic Motion
Converted from visualize_harmonic_motion.ipynb
"""

import os
import sys
import tempfile
import functools
from typing import Tuple
import argparse

import seaborn as sns
from absl import logging
import collections
import chex
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import frozen_dict
from flax.training import train_state
import optax
import ml_collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import matplotlib.ticker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_angle_networks.simulation import harmonic_motion_simulation
from action_angle_networks import models, train, analysis
from action_angle_networks.configs.harmonic_motion import (
    action_angle_flow, 
    action_angle_mlp, 
    euler_update_flow, 
    euler_update_mlp
)

PLT_STYLE_CONTEXT = ['science', 'ieee', 'grid']
matplotlib.rc("animation", html="jshtml")
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


def load_pretrained_model(workdir: str):
    """Load a pretrained model from workdir."""
    config, scaler, state, aux = analysis.load_from_workdir(workdir)
    return config, scaler, state, aux


def train_new_model(config_type: str = "action_angle_flow", output_dir: str = None):
    """Train a new model from scratch."""
    if config_type == "action_angle_flow":
        config = action_angle_flow.get_config()
    elif config_type == "action_angle_mlp":
        config = action_angle_mlp.get_config()
    elif config_type == "euler_update_flow":
        config = euler_update_flow.get_config()
    elif config_type == "euler_update_mlp":
        config = euler_update_mlp.get_config()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    workdir = output_dir or tempfile.mkdtemp()
    print(f"Training model with workdir: {workdir}")
    
    scaler, state, aux = train.train_and_evaluate(config, workdir)
    return config, scaler, state, aux


def plot_hamiltonian_changes(config, aux, output_dir: str = "output"):
    """Plot changes in Hamiltonian over training steps."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_test_metrics = aux["test"]["metrics"]
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    test_simulation_parameters = aux["test"]["simulation_parameters"]
    
    total_changes = {
        jump: np.asarray([
            all_test_metrics[step][jump]["mean_change_in_hamiltonians"] 
            for step in all_test_metrics
        ])
        for jump in config.test_time_jumps
    }
    steps = list(all_test_metrics.keys())
    
    # Compute actual Hamiltonian for reference
    actual_hamiltonian = harmonic_motion_simulation.compute_hamiltonian(
        test_positions[:1], test_momentums[:1], test_simulation_parameters
    )
    actual_hamiltonian = np.asarray(actual_hamiltonian).squeeze()
    
    try:
        with plt.style.context(PLT_STYLE_CONTEXT):
            fig, ax = plt.subplots()
            colors = plt.cm.viridis(np.linspace(0, 1, len(total_changes)))
            for jump_color, jump in zip(colors, config.test_time_jumps):
                total_changes_for_jump = total_changes[jump]
                ax.plot(steps, total_changes_for_jump, label=jump, color=jump_color)
            ax.axhline(y=actual_hamiltonian, c="gray", linestyle="--")
            ax.set_title("Mean Change in Hamiltonian")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Change")
            ax.set_yscale("log")
            ax.set_ylim(0.005, 300)
            ax.legend(title="Jump Size", loc="upper right", title_fontsize=6, fontsize=6)
            plt.savefig(f"{output_dir}/change_in_hamiltonian.pdf", dpi=300, bbox_inches="tight")
            plt.show()
    except Exception as e:
        plt.figure()
        plt.plot(steps, list(total_changes.values())[0])
        plt.title("Mean Change in Hamiltonian (simplified)")
        plt.xlabel("Steps")
        plt.ylabel("Change")
        plt.savefig(f"{output_dir}/change_in_hamiltonian_simple.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Used simplified plot due to: {e}")


def plot_test_loss(config, aux, output_dir: str = "output"):
    """Plot test loss over training steps."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_test_metrics = aux["test"]["metrics"]
    
    try:
        with plt.style.context(PLT_STYLE_CONTEXT):
            total_losses = {
                jump: np.asarray([
                    all_test_metrics[step][jump]["prediction_loss"] 
                    for step in all_test_metrics
                ])
                for jump in config.test_time_jumps
            }
            steps = list(all_test_metrics.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(total_losses)))

            fig, ax = plt.subplots()
            for jump_color, jump in zip(colors, config.test_time_jumps):
                total_losses_for_jump = total_losses[jump]
                ax.plot(steps, total_losses_for_jump, label=jump, color=jump_color)

            ax.set_title("Test Loss")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.legend(title="Jump Size")
            plt.savefig(f"{output_dir}/test_losses.pdf", dpi=300, bbox_inches="tight")
            plt.show()
    except Exception as e:
        print(f"Could not plot test loss: {e}")


def plot_action_angle_contours(config, state, scaler, aux, output_dir: str = "output"):
    """Plot action and angle contours in phase space."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_positions = aux["train"]["positions"]
    train_momentums = aux["train"]["momentums"]
    
    # Sample position-momentum space
    max_position = 1.2 * np.abs(train_positions).max()
    max_momentum = 1.2 * np.abs(train_momentums).max()
    plot_positions = jnp.linspace(-max_position, max_position, num=100)
    plot_momentums = jnp.linspace(-max_momentum, max_momentum, num=100)
    grid = jnp.meshgrid(plot_positions, plot_momentums)
    plot_positions = grid[0][:, :, jnp.newaxis]
    plot_momentums = grid[1][:, :, jnp.newaxis]
    
    # Pad coordinates
    trajectory_index = 0
    plot_positions = jnp.pad(
        plot_positions, ((0, 0), (0, 0), (trajectory_index, config.num_trajectories - trajectory_index - 1))
    )
    plot_momentums = jnp.pad(
        plot_momentums, ((0, 0), (0, 0), (trajectory_index, config.num_trajectories - trajectory_index - 1))
    )
    
    # Compute actions and angles
    try:
        _, _, auxiliary_predictions = jax.vmap(state.apply_fn, in_axes=(None, 0, 0, None))(
            state.params, plot_positions, plot_momentums, 0
        )
        plot_actions = auxiliary_predictions["actions"]
        plot_angles = auxiliary_predictions["current_angles"]
        
        # Rescale back to original data range
        plot_positions, plot_momentums = jax.vmap(train.inverse_transform_with_scaler, in_axes=(0, 0, None))(
            plot_positions, plot_momentums, scaler
        )
        train_positions_rescaled, train_momentums_rescaled = train.inverse_transform_with_scaler(
            train_positions, train_momentums, scaler
        )
        
        plot_positions, plot_momentums, plot_actions, plot_angles = jax.tree_map(
            lambda arr: arr[:, :, trajectory_index],
            (plot_positions, plot_momentums, plot_actions, plot_angles),
        )
        
        # Plot actions contour
        fig, ax = plt.subplots()
        contours = ax.contour(plot_positions, plot_momentums, plot_actions, 50, cmap="viridis")
        fig.colorbar(contours)
        ax.plot(
            train_positions_rescaled[:, trajectory_index],
            train_momentums_rescaled[:, trajectory_index],
            c="gray", linestyle="--"
        )
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        ax.set_title("Actions Contour")
        plt.savefig(f"{output_dir}/actions_contour.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        
        # Plot angles contour
        fig, ax = plt.subplots()
        contours = ax.contour(plot_positions, plot_momentums, plot_angles, 50, cmap="viridis")
        fig.colorbar(contours)
        ax.plot(
            train_positions_rescaled[:, trajectory_index],
            train_momentums_rescaled[:, trajectory_index],
            c="gray", linestyle="--"
        )
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        ax.set_title("Angles Contour")
        plt.savefig(f"{output_dir}/angles_contour.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        
    except Exception as e:
        print(f"Could not plot action-angle contours: {e}")


def plot_phase_space_trajectories(config, scaler, aux, output_dir: str = "output"):
    """Plot trajectories in phase space."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_positions = aux["train"]["positions"]
    train_momentums = aux["train"]["momentums"]
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    
    train_positions_rescaled, train_momentums_rescaled = train.inverse_transform_with_scaler(
        train_positions, train_momentums, scaler
    )
    test_positions_rescaled, test_momentums_rescaled = train.inverse_transform_with_scaler(
        test_positions, test_momentums, scaler
    )
    
    max_position = np.abs(train_positions_rescaled).max()
    max_momentum = np.abs(train_momentums_rescaled).max()
    
    try:
        with plt.style.context(PLT_STYLE_CONTEXT):
            harmonic_motion_simulation.static_plot_coordinates_in_phase_space(
                train_positions_rescaled,
                train_momentums_rescaled,
                title="Train Trajectories",
                max_position=max_position,
                max_momentum=max_momentum,
            )
            plt.savefig(f"{output_dir}/train_trajectories_phase_space.pdf", dpi=300, bbox_inches="tight")
            plt.show()
            
            harmonic_motion_simulation.static_plot_coordinates_in_phase_space(
                test_positions_rescaled,
                test_momentums_rescaled,
                title="Test Trajectories",
                max_position=max_position,
                max_momentum=max_momentum,
            )
            plt.savefig(f"{output_dir}/test_trajectories_phase_space.pdf", dpi=300, bbox_inches="tight")
            plt.show()
    except Exception as e:
        print(f"Could not plot phase space trajectories: {e}")


def predict_for_trajectory(state, scaler, config, positions_for_trajectory, momentums_for_trajectory, jump: int):
    """Make predictions for a trajectory with given time jump."""
    curr_positions, curr_momentums, target_positions, target_momentums = train.get_coordinates_for_time_jump(
        positions_for_trajectory, momentums_for_trajectory, jump
    )
    predicted_positions, predicted_momentums, auxiliary_predictions = train.compute_predictions(
        state, curr_positions, curr_momentums, jump * config.time_delta
    )
    predicted_positions, predicted_momentums = train.inverse_transform_with_scaler(
        predicted_positions, predicted_momentums, scaler
    )
    return predicted_positions, predicted_momentums


def plot_one_step_predictions(config, state, scaler, aux, output_dir: str = "output"):
    """Plot one-step predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_positions = aux["train"]["positions"]
    train_momentums = aux["train"]["momentums"]
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    
    train_positions_rescaled, train_momentums_rescaled = train.inverse_transform_with_scaler(
        train_positions, train_momentums, scaler
    )
    max_position = np.abs(train_positions_rescaled).max()
    max_momentum = np.abs(train_momentums_rescaled).max()
    
    jump = 1
    try:
        predicted_positions, predicted_momentums = predict_for_trajectory(
            state, scaler, config, train_positions, train_momentums, jump
        )
        
        with plt.style.context(PLT_STYLE_CONTEXT):
            harmonic_motion_simulation.static_plot_coordinates_in_phase_space(
                predicted_positions,
                predicted_momentums,
                title=f"Predicted Train Trajectories: Jump {jump}",
                max_position=max_position,
                max_momentum=max_momentum,
            )
            plt.savefig(f"{output_dir}/predicted_train_trajectories_jump_{jump}.pdf", dpi=300, bbox_inches="tight")
            plt.show()
    except Exception as e:
        print(f"Could not plot one-step predictions: {e}")


def plot_prediction_errors(config, state, scaler, aux, output_dir: str = "output"):
    """Plot prediction errors over time."""
    os.makedirs(output_dir, exist_ok=True)
    
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    test_simulation_parameters = aux["test"]["simulation_parameters"]
    
    jump = 1
    try:
        _, _, target_positions, target_momentums = train.get_coordinates_for_time_jump(
            test_positions, test_momentums, jump
        )
        predicted_positions, predicted_momentums = predict_for_trajectory(
            state, scaler, config, test_positions, test_momentums, jump
        )
        target_positions, target_momentums = train.inverse_transform_with_scaler(
            target_positions, target_momentums, scaler
        )
        
        errors = jax.vmap(train.compute_loss)(
            predicted_positions, predicted_momentums, target_positions, target_momentums
        )
        
        with plt.style.context(PLT_STYLE_CONTEXT):
            plt.figure(figsize=(10, 6))
            plt.plot(errors, c="teal")
            plt.grid(axis='x')
            plt.title("Prediction Error vs Time")
            plt.ylim((errors.min(), errors.max() * 1.2))
            plt.xlabel("Time")
            plt.ylabel("Prediction Error")
            plt.savefig(f"{output_dir}/prediction_errors.pdf", dpi=300, bbox_inches="tight")
            plt.show()
            
        # Plot Hamiltonian changes
        predicted_hamiltonians = jax.vmap(
            harmonic_motion_simulation.compute_hamiltonian, in_axes=(0, 0, None)
        )(predicted_positions, predicted_momentums, test_simulation_parameters)
        true_hamiltonians = jax.vmap(
            harmonic_motion_simulation.compute_hamiltonian, in_axes=(0, 0, None)
        )(target_positions, target_momentums, test_simulation_parameters)
        relative_delta_hamiltonians = (predicted_hamiltonians - true_hamiltonians) / true_hamiltonians
        
        with plt.style.context(PLT_STYLE_CONTEXT):
            plt.figure(figsize=(10, 6))
            plt.plot(relative_delta_hamiltonians, c="olivedrab")
            plt.grid(axis='x')
            plt.ylim((relative_delta_hamiltonians.min(), relative_delta_hamiltonians.max() * 1.2))
            plt.title("Change in Hamiltonian vs Time")
            plt.xlabel("Time")
            plt.ylabel("Relative Change in Hamiltonian")
            plt.savefig(f"{output_dir}/hamiltonian_changes.pdf", dpi=300, bbox_inches="tight")
            plt.show()
            
    except Exception as e:
        print(f"Could not plot prediction errors: {e}")


def plot_actions_distribution(config, state, aux, output_dir: str = "output"):
    """Plot distribution of actions and angular velocities."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_positions = aux["train"]["positions"]
    train_momentums = aux["train"]["momentums"]
    train_simulation_parameters = aux["train"]["simulation_parameters"]
    
    try:
        jump = 1
        curr_positions, curr_momentums, *_ = train.get_coordinates_for_time_jump(
            train_positions, train_momentums, jump
        )
        
        _, _, auxiliary_predictions = state.apply_fn(
            state.params, curr_positions, curr_momentums, 0
        )
        
        actions = auxiliary_predictions["actions"]
        angular_velocities = auxiliary_predictions["angular_velocities"]
        
        # Plot actions histogram
        plt.figure(figsize=(10, 6))
        for trajectory in range(actions.shape[1]):
            plt.hist(actions[:, trajectory], bins=50, alpha=0.7, label=f"Trajectory {trajectory}")
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.3f}"))
        plt.xlabel("Actions")
        plt.ylabel("Count")
        plt.title("Distribution of Actions")
        plt.legend()
        plt.savefig(f"{output_dir}/actions_distribution.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        
        # Plot angular velocities histogram
        true_angular_velocities = harmonic_motion_simulation.compute_normal_modes(train_simulation_parameters)[0]
        
        plt.figure(figsize=(10, 6))
        for trajectory in range(angular_velocities.shape[1]):
            plt.hist(angular_velocities[:, trajectory], bins=50, alpha=0.7, label=f"Trajectory {trajectory}")
            plt.axvline(x=true_angular_velocities[trajectory], linestyle='--', 
                       label=f"True ω_{trajectory}")
        plt.xlabel("Angular Frequency")
        plt.ylabel("Count")
        plt.title("Distribution of Angular Frequencies")
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.4f}"))
        plt.legend()
        plt.savefig(f"{output_dir}/angular_frequencies_distribution.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        
        print(f"Mean angular velocities: {angular_velocities.mean(axis=0)}")
        print(f"True angular velocities: {true_angular_velocities}")
        
    except Exception as e:
        print(f"Could not plot actions distribution: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Action-Angle Networks for Harmonic Motion")
    parser.add_argument("--mode", choices=["train", "load"], default="train",
                       help="Train new model or load pretrained model")
    parser.add_argument("--config", default="action_angle_flow",
                       choices=["action_angle_flow", "action_angle_mlp", "euler_update_flow", "euler_update_mlp"],
                       help="Configuration to use for training")
    parser.add_argument("--workdir", type=str, help="Working directory (for training or loading)")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for plots")
    parser.add_argument("--num_train_steps", type=int, default=1000,
                       help="Number of training steps (for quick testing)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Training new model...")
        config, scaler, state, aux = train_new_model(args.config, args.workdir)
        # Override training steps for quick testing
        config.num_train_steps = args.num_train_steps
    else:
        if not args.workdir:
            raise ValueError("Must specify --workdir when loading pretrained model")
        print(f"Loading pretrained model from {args.workdir}")
        config, scaler, state, aux = load_pretrained_model(args.workdir)
    
    print("Generating plots...")
    
    # Generate all plots
    plot_hamiltonian_changes(config, aux, args.output_dir)
    plot_test_loss(config, aux, args.output_dir)
    plot_action_angle_contours(config, state, scaler, aux, args.output_dir)
    plot_phase_space_trajectories(config, scaler, aux, args.output_dir)
    plot_one_step_predictions(config, state, scaler, aux, args.output_dir)
    plot_prediction_errors(config, state, scaler, aux, args.output_dir)
    plot_actions_distribution(config, state, aux, args.output_dir)
    
    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()