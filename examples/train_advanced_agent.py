import os
import sys
import numpy as np
import torch
import random
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.agents.advanced_agent import AdvancedAgent

def create_env(render_mode=None, team_size=1):
    """Create and configure the PyQuaticus environment."""
    env = PyQuaticusEnv(
        render_mode=render_mode,
        team_size=team_size,  # Now configurable team size
        config_dict={
            'max_time': 300,
            'max_score': 10,
            'sim_speedup_factor': 10,
            'render_agent_ids': True
        }
    )
    return env

def train():
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create environment with 1v1 setup
    logger.info("Creating environment...")
    env = create_env(render_mode=None, team_size=1)  # 1v1 setup
    
    # Get agent IDs generically (do not assume naming)
    agents = list(env.possible_agents) if hasattr(env, 'possible_agents') else []
    if not agents:
        raise ValueError("No agents found in the environment")
    blue_agent_id = agents[0]
    red_agent_id = agents[1] if len(agents) > 1 else None
    
    if red_agent_id is not None:
        logger.info(f"Primary agent: {blue_agent_id}, Opponent agent: {red_agent_id}")
    else:
        logger.info(f"Single-agent environment. Controlling: {blue_agent_id}")
    
    # Print environment information
    logger.info("Environment created successfully")
    logger.info(f"Possible agents: {env.possible_agents}")
    
    # Print action and observation spaces for each agent
    for agent_id in env.possible_agents:
        logger.info(f"Agent {agent_id} - Action space: {env.action_spaces[agent_id]}")
        logger.info(f"Agent {agent_id} - Observation space: {env.observation_spaces[agent_id]}")
    
    logger.info("Environment setup complete")
    
    if not hasattr(env, 'possible_agents') or not env.possible_agents:
        raise ValueError("No agents found in the environment")
    
    # Get state and action dimensions for blue agent
    state_size = env.observation_space(blue_agent_id).shape[0]
    action_size = env.action_space(blue_agent_id).n
    
    # Create blue agent (our learning agent)
    blue_agent = AdvancedAgent(
        agent_id=blue_agent_id,
        env=env,
        state_size=state_size,
        action_size=action_size,
        seed=seed
    )
    
    # Create a simple red agent (opponent) only if present
    if red_agent_id is not None:
        red_agent = AdvancedAgent(
            agent_id=red_agent_id,
            env=env,
            state_size=state_size,
            action_size=action_size,
            seed=seed+1  # Different seed for red agent
        )
    
    # Training parameters
    n_episodes = 200  # Total number of episodes
    max_t = 50  # Max steps per episode (reduced from 600)
    
    # Early stopping parameters
    patience = 10  # Stop if no improvement for this many episodes
    min_improvement = 0.1  # Minimum improvement to reset patience
    best_score = -float('inf')
    no_improvement_count = 0
    save_dir = 'checkpoints'
    
    # Create a timestamped directory for this run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('checkpoints', f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving checkpoints to: {os.path.abspath(save_dir)}")
    
    # Training loop
    scores = []
    scores_window = []
    logger.info("Starting training...")
    best_avg = -1e9
    
    for i_episode in range(1, n_episodes + 1):
        if no_improvement_count >= patience and i_episode > 10:  # Give it at least 10 episodes
            logger.info(f"\nEarly stopping at episode {i_episode} - no improvement for {patience} episodes")
            break
            
        episode_start_time = datetime.now()
        logger.info(f"\n=== Starting episode {i_episode}/{n_episodes} (max {max_t} steps) ===")
        try:
            # Reset the environment and get initial observation
            logger.info("Resetting environment...")
            reset_ret = env.reset()
            
            # Handle the reset return value
            if isinstance(reset_ret, tuple):
                obs_dict = reset_ret[0]
                reset_info = reset_ret[1] if len(reset_ret) > 1 else {}
            else:
                obs_dict = reset_ret
                reset_info = {}
            
            logger.info(f"Environment reset. Agents: {list(obs_dict.keys())}")
            
            # Initialize episode variables
            score = 0
            done = False
            t = 0
            
            # Per-episode epsilon (simple linear decay)
            eps = max(0.01, 1.0 - i_episode / n_episodes)

            while not done and t < max_t:
                # Get actions for both agents
                blue_action = blue_agent.act(obs_dict[blue_agent_id]['observation'], eps=eps)

                actions = {blue_agent_id: int(blue_action)}
                if red_agent_id is not None:
                    # Red agent uses a simple policy (e.g., random or high exploration)
                    red_eps = 0.9
                    red_action = red_agent.act(obs_dict[red_agent_id]['observation'], eps=red_eps)
                    actions[red_agent_id] = int(red_action)
                
                # Take a step in the environment with both agents' actions
                try:
                    step_return = env.step(actions)
                    
                    # Handle different return formats
                    if len(step_return) == 5:  # New format: obs, reward, done, truncated, info
                        next_obs_dict, rewards, dones, truncated, infos = step_return
                    elif len(step_return) == 4:  # Old format: obs, reward, done, info
                        next_obs_dict, rewards, dones, infos = step_return
                        truncated = {blue_agent_id: False, red_agent_id: False}
                    else:
                        raise ValueError(f"Unexpected step return format: {step_return}")
                    
                    # Process the step return for blue agent (our learning agent)
                    reward = rewards[blue_agent_id]
                    
                    # Log the reward and done status
                    logger.info(f"Processed reward: {reward}, done: {done}")
                    
                    # Handle different done formats
                    if isinstance(dones, dict):
                        # end if per-agent done or global done
                        done = bool(dones.get(blue_agent_id, False) or dones.get("__all__", False))
                    else:
                        done = bool(dones)  # Convert to bool if it's a scalar
                    
                    # also check truncated global flag if dict
                    if isinstance(truncated, dict) and truncated.get("__all__", False):
                        done = True
                    
                    # Get the next state for blue agent
                    if isinstance(next_obs_dict, dict):
                        if blue_agent_id in next_obs_dict:
                            agent_obs = next_obs_dict[blue_agent_id]
                            if isinstance(agent_obs, dict) and 'observation' in agent_obs:
                                next_state = agent_obs['observation']
                            else:
                                next_state = agent_obs
                        else:
                            # If agent_id not in dict, try to use the first value
                            first_key = next(iter(next_obs_dict))
                            agent_obs = next_obs_dict[first_key]
                            if isinstance(agent_obs, dict) and 'observation' in agent_obs:
                                next_state = agent_obs['observation']
                            else:
                                next_state = agent_obs
                    else:
                        # If not a dict, use as is
                        next_state = next_obs_dict
                        
                    # Store the experience in the blue agent's replay buffer
                    blue_agent.step(
                        obs_dict[blue_agent_id]['observation'],
                        blue_action,
                        reward,
                        next_state,
                        done
                    )
                    
                    # Update the target network
                    if len(blue_agent.memory) > blue_agent.batch_size:
                        blue_agent.learn(blue_agent.gamma)
                    
                    # Update the score and observation
                    score += reward
                    obs_dict = next_obs_dict
                    t += 1
                    
                except Exception as e:
                    logger.error(f"Error in env.step(): {str(e)}", exc_info=True)
                    raise
            
            # Save episode score
            scores_window.append(score)
            scores.append(score)

            # Episode summary and moving average (last 5)
            recent_avg = np.mean(scores[-5:]) if len(scores) >= 5 else score
            if recent_avg > best_score + min_improvement:
                best_score = recent_avg
                no_improvement_count = 0
                best_path = os.path.join(save_dir, 'best.pth')
                blue_agent.save(best_path)
                logger.info(f"New best average score: {best_score:.2f}, saved to {best_path}")
            else:
                no_improvement_count += 1
        
            # Log episode summary
            duration = (datetime.now() - episode_start_time).total_seconds()
            logger.info(f"Episode {i_episode} | Score: {score:.2f} | Recent Avg: {recent_avg:.2f} | Best: {best_score:.2f} | Steps: {t}")
            logger.info(f"Time: {duration:.1f}s | Epsilon: {eps:.3f} | No improvement: {no_improvement_count}/{patience}")
        
            # Save checkpoint every 10 episodes
            if i_episode % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{i_episode}.pth')
                blue_agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
                # Log memory stats
                if hasattr(blue_agent, 'memory') and hasattr(blue_agent.memory, '__len__'):
                    logger.info(f"Replay buffer size: {len(blue_agent.memory)}")
                
                # Log learning rate if using a scheduler
                if hasattr(blue_agent, 'scheduler') and blue_agent.scheduler is not None:
                    logger.info(f"Learning rate: {blue_agent.scheduler.get_last_lr()[0]:.6f}")
                
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            agent.save(os.path.join(save_dir, 'interrupted.pth'))
            break
        except Exception as e:
            print(f"Error during training: {e}")
            continue
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model.pth')
    agent.save(final_path)
    print(f'Training complete. Final model saved to {final_path}')
    return scores

if __name__ == "__main__":
    train()
