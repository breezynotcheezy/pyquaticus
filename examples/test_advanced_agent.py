import os
import sys
import numpy as np
import torch
import random

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.agents.advanced_agent import AdvancedAgent

def create_env(render_mode='human'):
    """Create and configure the PyQuaticus environment."""
    env = PyQuaticusEnv(
        render_mode=render_mode,
        team_size=1,
        config_dict={
            'max_time': 300,
            'max_score': 10,
            'sim_speedup_factor': 10,
            'render_agent_ids': True
        }
    )
    return env

def test(blue_model_path, red_model_path=None):
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = create_env(render_mode='human')
    
    # Determine agent IDs
    agents = list(env.possible_agents)
    if len(agents) < 2:
        raise ValueError("Expected at least 2 agents in the environment for 1v1")
    blue_id, red_id = agents[0], agents[1]
    
    # Get state and action dimensions
    state_size = env.observation_space(blue_id).shape[0]
    action_size = env.action_space(blue_id).n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agents
    blue = AdvancedAgent(
        agent_id=blue_id,
        env=env,
        state_size=state_size,
        action_size=action_size,
        seed=seed
    )
    red = AdvancedAgent(
        agent_id=red_id,
        env=env,
        state_size=state_size,
        action_size=action_size,
        seed=seed+1
    )
    
    # Load models
    blue.load(blue_model_path)
    print(f"Loaded BLUE model from {blue_model_path}")
    if red_model_path is not None and os.path.exists(red_model_path):
        red.load(red_model_path)
        print(f"Loaded RED model from {red_model_path}")
    else:
        # If no red path provided, mirror blue weights for a fair match
        red.load(blue_model_path)
        print(f"Loaded RED model from BLUE path (mirrored): {blue_model_path}")
    
    # Testing parameters
    n_episodes = 5
    max_t = 600
    
    print("Starting testing...")
    
    for i_episode in range(1, n_episodes + 1):
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs_dict = reset_ret[0]
        else:
            obs_dict = reset_ret
        score_blue = 0.0
        score_red = 0.0
        done = False
        
        for t in range(max_t):
            # Select actions (greedy)
            blue_state = obs_dict[blue_id]['observation'] if isinstance(obs_dict[blue_id], dict) and 'observation' in obs_dict[blue_id] else obs_dict[blue_id]
            red_state = obs_dict[red_id]['observation'] if isinstance(obs_dict[red_id], dict) and 'observation' in obs_dict[red_id] else obs_dict[red_id]
            a_blue = int(blue.act(blue_state, eps=0.0))
            a_red = int(red.act(red_state, eps=0.0))

            # Step environment with both actions
            step_return = env.step({blue_id: a_blue, red_id: a_red})
            
            # Handle different return formats
            if len(step_return) == 5:  # New format: obs, reward, done, truncated, info
                next_obs_dict, rewards, dones, truncated, infos = step_return
            else:  # Old format: obs, reward, done, info
                next_obs_dict, rewards, dones, infos = step_return
                truncated = {agent_id: False}
            
            # Rewards and done
            r_blue = float(rewards.get(blue_id, 0.0)) if isinstance(rewards, dict) else float(rewards)
            r_red = float(rewards.get(red_id, 0.0)) if isinstance(rewards, dict) else 0.0
            score_blue += r_blue
            score_red += r_red
            if isinstance(dones, dict):
                done = bool(dones.get("__all__", False) or dones.get(blue_id, False) or dones.get(red_id, False))
            else:
                done = bool(dones)
            if isinstance(truncated, dict) and truncated.get("__all__", False):
                done = True

            # Next obs
            obs_dict = next_obs_dict
            
            if done:
                break
        
        print(f'Episode {i_episode}\tBlue: {score_blue:.2f}\tRed: {score_red:.2f}')
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained AdvancedAgents (self-play)')
    parser.add_argument('--blue', type=str, default='checkpoints/final_model_blue.pth', help='Path to BLUE model')
    parser.add_argument('--red', type=str, default=None, help='Path to RED model (optional; mirrors blue if not set)')
    args = parser.parse_args()
    test(args.blue, args.red)
