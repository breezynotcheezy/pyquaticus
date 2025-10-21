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

def test(model_path):
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = create_env(render_mode='human')
    
    # Get agent ID (first agent in the environment)
    agent_id = env.possible_agents[0]
    
    # Get state and action dimensions
    state_size = env.observation_space(agent_id).shape[0]
    action_size = env.action_space(agent_id).n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent
    agent = AdvancedAgent(
        agent_id=agent_id,
        env=env,
        state_size=state_size,
        action_size=action_size,
        seed=seed
    )
    
    # Load the trained model
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Testing parameters
    n_episodes = 10
    max_t = 1000
    
    print("Starting testing...")
    
    for i_episode in range(1, n_episodes + 1):
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs_dict = reset_ret[0]
        else:
            obs_dict = reset_ret
        # Extract initial state for this agent
        if isinstance(obs_dict, dict) and agent_id in obs_dict:
            ag_obs = obs_dict[agent_id]
            if isinstance(ag_obs, dict) and 'observation' in ag_obs:
                state = ag_obs['observation']
            else:
                state = ag_obs
        else:
            state = np.array(obs_dict, dtype=np.float32).flatten()
        score = 0
        done = False
        
        for t in range(max_t):
            # Select action using the agent's policy (with epsilon=0 for greedy action selection)
            action = agent.act(state, eps=0.0)
            
            # Take action in environment
            step_return = env.step({agent_id: int(action)})
            
            # Handle different return formats
            if len(step_return) == 5:  # New format: obs, reward, done, truncated, info
                next_obs_dict, rewards, dones, truncated, infos = step_return
            else:  # Old format: obs, reward, done, info
                next_obs_dict, rewards, dones, infos = step_return
                truncated = {agent_id: False}
            
            # Extract next_state
            if isinstance(next_obs_dict, dict) and agent_id in next_obs_dict:
                ag_obs = next_obs_dict[agent_id]
                if isinstance(ag_obs, dict) and 'observation' in ag_obs:
                    next_state = ag_obs['observation']
                else:
                    next_state = ag_obs
            else:
                next_state = next_obs_dict
            next_state = np.array(next_state, dtype=np.float32).flatten()

            # Reward and done
            reward = float(rewards.get(agent_id, 0.0)) if isinstance(rewards, dict) else float(rewards)
            if isinstance(dones, dict):
                done = bool(dones.get(agent_id, False) or dones.get("__all__", False))
            else:
                done = bool(dones)
            if isinstance(truncated, dict) and truncated.get("__all__", False):
                done = True
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
        
        print(f'Episode {i_episode}\tScore: {score:.2f}')
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the trained AdvancedAgent')
    parser.add_argument('--model', type=str, default='checkpoints/final_model.pth',
                       help='Path to the trained model')
    
    args = parser.parse_args()
    test(args.model)
