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

    # Reward shaping: encourage progress toward enemy flag, penalize OOB/tagged, reward grabs/tags, small step penalty
    def shaped_reward(agent_id, team, agents, agent_inds_of_team, state, prev_state, **kwargs):
        try:
            team_idx = int(team)
            other_team_idx = int(not team_idx)
            if isinstance(agents, (list, tuple)):
                i = list(agents).index(agent_id) if agent_id in agents else 0
            elif isinstance(agents, dict):
                i = list(agents.keys()).index(agent_id) if agent_id in agents else 0
            else:
                i = 0
            pos = state["agent_position"][i]
            prev_pos = prev_state["agent_position"][i] if prev_state is not None else pos
            enemy_flag_pos = state["flag_home"][other_team_idx]
            # distance progress toward enemy flag
            d_prev = np.linalg.norm(prev_pos - enemy_flag_pos)
            d_now = np.linalg.norm(pos - enemy_flag_pos)
            progress = (d_prev - d_now)  # positive when moving closer
            # penalties
            oob_pen = -0.05 if state.get("agent_oob", [False]* (i+1))[i] else 0.0
            tagged_pen = -0.02 if state.get("agent_is_tagged", [False]* (i+1))[i] else 0.0
            # sparse bonuses
            if prev_state is not None:
                grab_bonus = 1.0 if state.get("grabs", [0,0])[team_idx] > prev_state.get("grabs", [0,0])[team_idx] else 0.0
                tag_bonus = 0.5 if state.get("tags", [0,0])[team_idx] > prev_state.get("tags", [0,0])[team_idx] else 0.0
            else:
                grab_bonus = 0.0
                tag_bonus = 0.0
            # small living penalty
            step_pen = -0.001
            return 0.2*progress + oob_pen + tagged_pen + grab_bonus + tag_bonus + step_pen
        except Exception:
            return 0.0

    # Wire shaped reward per agent if env exposes reward_config
    if hasattr(env, 'reward_config'):
        try:
            # Prefer explicit player IDs if available
            target_ids = list(getattr(env, 'players', {}).keys()) if hasattr(env, 'players') else list(env.possible_agents)
            for aid in target_ids:
                env.reward_config[aid] = shaped_reward
            logger.info(f"Wired shaped_reward for agents: {target_ids}")
        except Exception as e:
            logger.warning(f"Failed to wire shaped rewards: {e}")

    # Helper: map agent_id like 'agent_0' -> 0; fallback to 0
    def agent_index(aid):
        try:
            if isinstance(aid, (int, np.integer)):
                return int(aid)
            if isinstance(aid, str) and '_' in aid:
                return int(aid.split('_')[-1])
        except Exception:
            pass
        return 0

    # Helper: get observation for an agent from obs container
    def get_obs(obs_container, aid):
        if isinstance(obs_container, dict):
            if aid in obs_container:
                val = obs_container[aid]
            else:
                # fallback to first value
                val = next(iter(obs_container.values()))
        else:
            idx = agent_index(aid)
            val = obs_container[idx]
        if isinstance(val, dict) and 'observation' in val:
            return val['observation']
        return val

    # Helper: get per-agent value from dict or array-like
    def get_val(container, aid, default=0.0):
        if isinstance(container, dict):
            return container.get(aid, default)
        try:
            return container[agent_index(aid)]
        except Exception:
            return default
    
    if not hasattr(env, 'possible_agents') or not env.possible_agents:
        raise ValueError("No agents found in the environment")
    
    # Get state and action dimensions for blue agent
    state_size = env.observation_space(blue_agent_id).shape[0]
    action_size = env.action_space(blue_agent_id).n
    
    # Create blue agent (learning agent)
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
    n_episodes = 400  # Total number of episodes
    max_t = 600  # Steps per episode (increase for better play)
    
    # Early stopping parameters
    patience = 20  # Stop if no improvement for this many episodes
    min_improvement = 0.05  # Minimum improvement to reset patience
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
            
            # Per-episode epsilons (decay for both agents)
            eps_blue = max(0.05, 1.0 - i_episode / n_episodes)
            eps_red = max(0.05, 1.0 - i_episode / n_episodes)

            while not done and t < max_t:
                # Get actions for both agents (self-play)
                blue_state = get_obs(obs_dict, blue_agent_id)
                blue_action = blue_agent.act(blue_state, eps=eps_blue)

                actions = {blue_agent_id: int(blue_action)}
                if red_agent_id is not None:
                    red_state = get_obs(obs_dict, red_agent_id)
                    red_action = red_agent.act(red_state, eps=eps_red)
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
                    
                    # Process rewards for both agents
                    reward_blue = float(get_val(rewards, blue_agent_id, 0.0))
                    reward_red = float(get_val(rewards, red_agent_id, 0.0)) if red_agent_id is not None else 0.0
                    
                    # Handle different done formats
                    if isinstance(dones, dict):
                        # end if per-agent done or global done
                        done = bool(get_val(dones, blue_agent_id, False) or dones.get("__all__", False))
                    else:
                        done = bool(dones)  # Convert to bool if it's a scalar
                    
                    # also check truncated global flag if dict
                    if isinstance(truncated, dict) and truncated.get("__all__", False):
                        done = True
                    
                    # Next states
                    next_state_blue = get_obs(next_obs_dict, blue_agent_id)
                    next_state_red = get_obs(next_obs_dict, red_agent_id) if red_agent_id is not None else None

                    # Store experiences
                    blue_agent.step(
                        blue_state,
                        blue_action,
                        reward_blue,
                        next_state_blue,
                        done
                    )
                    if red_agent_id is not None:
                        red_agent.step(
                            red_state,
                            red_action,
                            reward_red,
                            next_state_red,
                            done
                        )

                    # Learn updates
                    if len(blue_agent.memory) > blue_agent.batch_size:
                        # Sample experiences and learn
                        experiences = blue_agent.memory.sample()
                        if experiences is not None:
                            blue_agent.learn(experiences)
                    
                    if red_agent_id is not None and len(red_agent.memory) > red_agent.batch_size:
                        # Sample experiences and learn
                        experiences = red_agent.memory.sample()
                        if experiences is not None:
                            red_agent.learn(experiences)

                    # Update score (track blue by default) and advance
                    score += reward_blue
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
                best_blue = os.path.join(save_dir, 'best_blue.pth')
                blue_agent.save(best_blue)
                if red_agent_id is not None:
                    best_red = os.path.join(save_dir, 'best_red.pth')
                    red_agent.save(best_red)
                logger.info(f"New best average score: {best_score:.2f}, saved blue->{best_blue}{' red->'+best_red if red_agent_id is not None else ''}")
            else:
                no_improvement_count += 1
        
            # Log episode summary
            duration = (datetime.now() - episode_start_time).total_seconds()
            logger.info(f"Episode {i_episode} | Score: {score:.2f} | Recent Avg: {recent_avg:.2f} | Best: {best_score:.2f} | Steps: {t}")
            logger.info(f"Time: {duration:.1f}s | Eps blue: {eps_blue:.3f} | Eps red: {eps_red:.3f} | No improvement: {no_improvement_count}/{patience}")
        
            # Save checkpoint every 10 episodes (both agents)
            if i_episode % 10 == 0:
                checkpoint_blue = os.path.join(save_dir, f'checkpoint_blue_{i_episode}.pth')
                blue_agent.save(checkpoint_blue)
                if red_agent_id is not None:
                    checkpoint_red = os.path.join(save_dir, f'checkpoint_red_{i_episode}.pth')
                    red_agent.save(checkpoint_red)
                logger.info(f"Saved checkpoints: blue->{checkpoint_blue}{' red->'+checkpoint_red if red_agent_id is not None else ''}")
            
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
    
    # Save final models
    final_blue = os.path.join(save_dir, 'final_model_blue.pth')
    blue_agent.save(final_blue)
    if red_agent_id is not None:
        final_red = os.path.join(save_dir, 'final_model_red.pth')
        red_agent.save(final_red)
        print(f'Training complete. Final models saved to {final_blue} and {final_red}')
    else:
        print(f'Training complete. Final model saved to {final_blue}')
    return scores

if __name__ == "__main__":
    train()
