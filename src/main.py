#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import create_dataloader
from train import OT_DBC, Diffusion_BC, OT_BC, train_model
from evaluate import DummyEnv, evaluate_policy_ood

def experiment_loss_ablation():
    print("\nRunning Experiment 1: Loss Ablation Study")
    dataloader = create_dataloader(num_samples=500, state_dim=10, action_dim=2, batch_size=32)
    num_epochs = 5
    
    model_ot_dbc = OT_DBC(state_dim=10, action_dim=2, hidden_dim=32, ot_lambda=1.0)
    model_ot_dbc, loss_hist_ot_dbc = train_model(model_ot_dbc, dataloader, num_epochs=num_epochs)
    
    model_diff_bc = Diffusion_BC(state_dim=10, action_dim=2, hidden_dim=32)
    model_diff_bc, loss_hist_diff_bc = train_model(model_diff_bc, dataloader, num_epochs=num_epochs)
    
    model_ot_bc = OT_BC(state_dim=10, action_dim=2, hidden_dim=32, ot_lambda=1.0)
    model_ot_bc, loss_hist_ot_bc = train_model(model_ot_bc, dataloader, num_epochs=num_epochs)
    
    epochs = np.arange(1, num_epochs+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss_hist_ot_dbc, 'o-', label="OT-DBC")
    plt.plot(epochs, loss_hist_diff_bc, 's-', label="Diffusion_BC (No OT)")
    plt.plot(epochs, loss_hist_ot_bc, '^-', label="OT_BC (No Diffusion)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss (Loss Ablation Study)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/training_loss_lossAblation.pdf", bbox_inches="tight")
    plt.close()
    
    print("Experiment 1 complete. Loss curves saved as training_loss_lossAblation.pdf\n")
    return model_ot_dbc, model_diff_bc, model_ot_bc

def experiment_hyperparameter_sensitivity():
    print("\nRunning Experiment 2: Hyperparameter Sensitivity on OT Loss Regularization")
    dataloader = create_dataloader(num_samples=400, state_dim=10, action_dim=2, batch_size=32)
    num_epochs = 5
    
    lambda_values = [0.1, 0.5, 1.0, 2.0]
    performance_results = {}
    
    for lam in lambda_values:
        print(f"\nTraining OT-DBC model with ot_lambda = {lam}")
        model = OT_DBC(state_dim=10, action_dim=2, hidden_dim=32, ot_lambda=lam)
        _, loss_history = train_model(model, dataloader, num_epochs=num_epochs)
        final_loss = loss_history[-1]
        performance_results[lam] = final_loss
        print(f"Final average loss for lambda={lam}: {final_loss:.4f}")
    
    plt.figure(figsize=(6,4))
    lambdas = list(performance_results.keys())
    losses = [performance_results[lam] for lam in lambdas]
    sns.barplot(x=lambdas, y=losses, palette="viridis")
    plt.xlabel("ot_lambda")
    plt.ylabel("Final Average Loss")
    plt.title("Hyperparameter Sensitivity: OT Loss Weight")
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/hyperparameterSensitivity_otLoss.pdf", bbox_inches="tight")
    plt.close()
    
    print("Experiment 2 complete. Hyperparameter sensitivity plot saved as hyperparameterSensitivity_otLoss.pdf\n")
    return performance_results

def experiment_ood_generalization(model_ot_dbc, model_baseline):
    print("\nRunning Experiment 3: OOD Generalization Study")
    env = DummyEnv(state_dim=10)
    
    print("\nEvaluating OT-DBC model under OOD conditions:")
    avg_reward_ot_dbc = evaluate_policy_ood(model_ot_dbc, env, num_episodes=5, noise_std=0.1)
    
    print("\nEvaluating Baseline Diffusion_BC model under OOD conditions:")
    avg_reward_baseline = evaluate_policy_ood(model_baseline, env, num_episodes=5, noise_std=0.1)
    
    models = ["OT-DBC", "Baseline (Diffusion_BC)"]
    rewards = [avg_reward_ot_dbc, avg_reward_baseline]
    plt.figure(figsize=(6,4))
    sns.barplot(x=models, y=rewards, palette="coolwarm")
    plt.ylabel("Average OOD Cumulative Reward")
    plt.title("OOD Performance Comparison")
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/ood_inferenceLatency.pdf", bbox_inches="tight")
    plt.close()
    
    print("Experiment 3 complete. OOD performance plot saved as ood_inferenceLatency.pdf\n")
    return avg_reward_ot_dbc, avg_reward_baseline

def test_all():
    print("Starting quick test of all experiments:")
    
    model_ot_dbc, model_diff_bc, model_ot_bc = experiment_loss_ablation()
    _ = experiment_hyperparameter_sensitivity()
    _ = experiment_ood_generalization(model_ot_dbc, model_diff_bc)
    
    print("Quick test completed successfully.")
    print("Setting status_enum to 'stopped'")
    status_enum = "stopped"

if __name__ == "__main__":
    test_all()
