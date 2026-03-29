"""
Reinforcement Learning training script using PPO on surrogate environment.
Trains a policy on the learned dynamics model.
"""
import argparse
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import torch

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import RL_CONFIG, get_policy_path, STATE_DIM, ACTION_DIM, HF_PARAMS
from surrogate_env import SurrogateEnv
from logger import ExperimentLogger


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================

class AdaptiveTimeSteppingCallback(BaseCallback):
    """
    Dynamically adjusts the environment's dt based on the surrogate's 
    estimated Lipschitz constant to maintain the theoretical HJB error bound:
    dt <= sqrt(eta) / ((1 - gamma) * L_f)
    """
    def __init__(self, env, gamma=0.99, verbose=0):
        super().__init__(verbose)
        # Handle DummyVecEnv wrapping
        self.target_env = env.envs[0] if hasattr(env, 'envs') else env
        # Unpack Monitor if present
        if hasattr(self.target_env, 'env'):
            self.target_env = self.target_env.env
            
        self.gamma = gamma
        self.eta = 0.01  # Default trust region / KL penalty approximation
        self.base_dt = HF_PARAMS['DT']
        
    def _estimate_lipschitz(self) -> float:
        """Estimate the maximum Lipschitz constant L_f of the surrogate models."""
        # We estimate L_f by calculating the spectral norm of the Jacobian 
        # for a batch of random states.
        
        # We need the surrogate models from the environment
        if not hasattr(self.target_env, 'models') or len(self.target_env.models) == 0:
            return 1.0 # fallback
            
        models = self.target_env.models
        device = self.target_env.device
        
        # Sample random states and actions
        B = 32
        s = torch.randn(B, STATE_DIM, device=device, requires_grad=True)
        a = torch.randn(B, ACTION_DIM, device=device) * 2 - 1
        x = torch.cat([s, a], dim=-1)
        
        # Use the first ensemble model for estimation
        model = models[0]
        model.eval()
        f = model(x)
        
        # Compute Jacobian J = df/ds
        J = torch.zeros(B, STATE_DIM, STATE_DIM, device=device)
        for i in range(STATE_DIM):
            v = torch.zeros_like(f)
            v[:, i] = 1.0
            grad_s = torch.autograd.grad(f, s, grad_outputs=v, create_graph=False, retain_graph=True)[0]
            J[:, i, :] = grad_s
            
        # Estimate spectral norm (max singular value)
        # J_np = J.detach().cpu().numpy()
        # To keep it simple in torch without SVD:
        # L_f is bounded by the Frobenius norm
        norms = torch.linalg.matrix_norm(J, ord='fro')
        return norms.max().item()

    def _on_step(self) -> bool:
        # Adjust dt every 2048 steps (typically one PPO rollout)
        if self.n_calls % 2048 == 0:
            L_f = self._estimate_lipschitz()
            
            # The theoretical step-size rule
            # Add a small epsilon to L_f to avoid division by zero
            ideal_dt = np.sqrt(self.eta) / ((1.0 - self.gamma) * (L_f + 1e-6))
            
            # Bound the dynamic dt to avoid extreme simulation failures
            # It shouldn't be much larger than base_dt, but can be smaller
            new_dt = np.clip(ideal_dt, 0.001, self.base_dt * 2.0)
            
            # Inject the new dt into the environment
            # Note: We patch it into HF_PARAMS dict globally so env.step picks it up, 
            # or directly onto the env if we refactored it.
            # For now, HF_PARAMS['DT'] is what SurrogateEnv uses inside step().
            HF_PARAMS['DT'] = new_dt
            
            if self.verbose > 0:
                print(f"   [Adaptive DT] L_f: {L_f:.3f} | Ideal dt: {ideal_dt:.4f} | Applied dt: {new_dt:.4f}")
                
            # Log it if logger callback exists
            if hasattr(self, 'logger_cb') and self.logger_cb is not None:
                self.logger_cb.logger.log_metrics({
                    'rl/adaptive_dt': new_dt,
                    'rl/lipschitz_Lf': L_f
                }, step=self.n_calls)
                
        return True


# ============================================================================
# RL TRAINING
# ============================================================================

def train_rl_policy(
    env,
    config: dict,
    save_path: Path,
    logger: ExperimentLogger = None,
    verbose: int = 1
):
    """Train PPO policy on surrogate environment.
    
    Args:
        env: Surrogate environment
        config: RL configuration dictionary
        save_path: Path to save trained policy
        logger: Experiment logger
        verbose: Verbosity level
        
    Returns:
        Trained PPO model
    """
    print("\n🚀 Starting RL training...")
    print(f"   Algorithm: {config['algorithm']}")
    print(f"   Policy: {config['policy']}")
    print(f"   Total timesteps: {config['total_timesteps']}")
    print(f"   Learning rate: {config['learning_rate']}")
    
    # Create PPO model
    model = PPO(
        policy=config['policy'],
        env=env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        verbose=verbose,
        tensorboard_log='./runs/' if logger is None else None
    )
    
    print(f"\n🏗️  Model architecture:")
    print(f"   Policy network: {config['policy']}")
    print(f"   Total parameters: {sum(p.numel() for p in model.policy.parameters())}")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback (save every N steps)
    checkpoint_dir = save_path.parent / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix='rl_model'
    )
    callbacks.append(checkpoint_callback)
    
    # Custom logging callback
    logger_cb = None
    if logger is not None:
        from stable_baselines3.common.callbacks import BaseCallback
        
        class LoggerCallback(BaseCallback):
            def __init__(self, logger, verbose=0):
                super().__init__(verbose)
                self.logger = logger
            
            def _on_step(self) -> bool:
                # Log every 100 steps
                if self.n_calls % 100 == 0:
                    if len(self.model.ep_info_buffer) > 0:
                        mean_reward = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                        mean_length = sum(ep['l'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                        
                        self.logger.log_metrics({
                            'rl/mean_episode_reward': mean_reward,
                            'rl/mean_episode_length': mean_length,
                            'rl/timesteps': self.n_calls
                        }, step=self.n_calls)
                return True
        
        logger_cb = LoggerCallback(logger)
        callbacks.append(logger_cb)

    # Adaptive Time-Stepping Callback
    adaptive_dt_cb = AdaptiveTimeSteppingCallback(env, gamma=config['gamma'], verbose=verbose)
    adaptive_dt_cb.logger_cb = logger_cb
    callbacks.append(adaptive_dt_cb)
    
    # Train model
    print("\n🎯 Training...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )
    
    print(f"\n✓ Training complete!")
    
    # Save model
    print(f"\n💾 Saving model to: {save_path}")
    model.save(save_path)
    
    if logger is not None:
        logger.log_model(save_path)
    
    return model


# ============================================================================
# POLICY EVALUATION
# ============================================================================

def evaluate_policy(
    model,
    env,
    n_episodes: int = 10,
    deterministic: bool = True
):
    """Evaluate trained policy.
    
    Args:
        model: Trained model
        env: Environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        
    Returns:
        Dictionary with evaluation statistics
    """
    print(f"\n📊 Evaluating policy over {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 5 == 0:
            print(f"   Episode {episode+1}/{n_episodes} | "
                  f"Reward: {episode_reward:.3f} | "
                  f"Length: {episode_length}")
    
    import numpy as np
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }
    
    print(f"\n✓ Evaluation complete!")
    print(f"   Mean reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"   Reward range: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
    print(f"   Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    
    return stats


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train RL policy on surrogate environment'
    )
    parser.add_argument(
        '--surrogate_path',
        type=str,
        required=True,
        help='Path to trained surrogate model'
    )
    parser.add_argument(
        '--state_scaler_path',
        type=str,
        required=True,
        help='Path to state scaler'
    )
    parser.add_argument(
        '--action_scaler_path',
        type=str,
        required=True,
        help='Path to action scaler'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save trained policy (e.g., models/policy_v1.zip)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='Path to custom config file (optional)'
    )
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=None,
        help='Override total timesteps'
    )
    parser.add_argument(
        '--log_experiment',
        action='store_true',
        help='Log to WandB/TensorBoard'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate policy after training'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RL POLICY TRAINING")
    print("=" * 70)
    
    # Load configuration
    config = RL_CONFIG.copy()
    if args.total_timesteps is not None:
        config['total_timesteps'] = args.total_timesteps
    
    # Initialize logger
    logger = None
    if args.log_experiment:
        logger = ExperimentLogger(
            experiment_name='rl_training',
            config=config
        )
    
    # Create surrogate environment
    print(f"\n🌍 Creating surrogate environment...")
    env = SurrogateEnv(
        surrogate_model_path=Path(args.surrogate_path),
        state_scaler_path=Path(args.state_scaler_path),
        action_scaler_path=Path(args.action_scaler_path)
    )
    
    # Wrap with Monitor
    env = Monitor(env)
    
    print(f"   Environment created successfully")
    
    # Train policy
    model = train_rl_policy(
        env=env,
        config=config,
        save_path=Path(args.save_path),
        logger=logger,
        verbose=1
    )
    
    # Evaluate if requested
    if args.evaluate:
        stats = evaluate_policy(model, env, n_episodes=20)
        
        if logger is not None:
            logger.log_metrics({
                'eval/mean_reward': stats['mean_reward'],
                'eval/std_reward': stats['std_reward'],
                'eval/mean_length': stats['mean_length']
            })
    
    # Cleanup
    if logger is not None:
        logger.finish()
    
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ RL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Policy saved: {args.save_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
