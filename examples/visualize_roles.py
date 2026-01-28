# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
Simple visualization script to observe hierarchical role assignment.

This script runs a 3v3 game with random agents and prints role changes
to the console so you can see how the role system works.
"""

import argparse
import numpy as np
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK, DEFEND, INTERCEPT, ROLE_NAMES
import logging

def extract_role_from_obs(observation):
    """Extract role ID from observation."""
    try:
        if observation is None:
            return ATTACK
        obs_array = np.asarray(observation)
        if len(obs_array.shape) == 1 and obs_array.shape[0] >= 3:
            role_one_hot = obs_array[-3:]
            role_id = np.argmax(role_one_hot)
            if role_id in [ATTACK, DEFEND, INTERCEPT]:
                return role_id
        return ATTACK
    except Exception:
        return ATTACK

def print_roles(roles, step):
    """Print current roles in a formatted way."""
    print(f"\n=== Step {step} ===")
    
    # Group by team
    blue_agents = []
    red_agents = []
    
    for agent_id, role_id in roles.items():
        if agent_id.startswith('agent_'):
            agent_num = int(agent_id.split('_')[1])
            if agent_num < 3:
                blue_agents.append((agent_id, role_id))
            else:
                red_agents.append((agent_id, role_id))
    
    # Print blue team
    print("BLUE TEAM:")
    for agent_id, role_id in blue_agents:
        role_name = ROLE_NAMES[role_id]
        print(f"  {agent_id}: {role_name}")
    
    # Print red team  
    print("RED TEAM:")
    for agent_id, role_id in red_agents:
        role_name = ROLE_NAMES[role_id]
        print(f"  {agent_id}: {role_name}")

def analyze_game_state(env):
    """Analyze and print current game state."""
    try:
        if hasattr(env, 'par_env') and hasattr(env.par_env, 'state'):
            state = env.par_env.state
            
            print("\n--- Game State ---")
            
            # Flag status
            if 'team_has_flag' in state:
                blue_has_flag = state['team_has_flag'][Team.BLUE_TEAM.value]
                red_has_flag = state['team_has_flag'][Team.RED_TEAM.value]
                print(f"Blue has enemy flag: {blue_has_flag}")
                print(f"Red has enemy flag: {red_has_flag}")
            
            # Agent flag status
            if 'agent_has_flag' in state:
                for i, has_flag in enumerate(state['agent_has_flag']):
                    if has_flag:
                        agent_id = f"agent_{i}"
                        team = "Blue" if i < 3 else "Red"
                        print(f"{agent_id} ({team}) is carrying a flag!")
            
            # Scores
            if 'captures' in state:
                blue_score = state['captures'][Team.BLUE_TEAM.value]
                red_score = state['captures'][Team.RED_TEAM.value]
                print(f"Score - Blue: {blue_score}, Red: {red_score}")
                
    except Exception as e:
        print(f"Error analyzing game state: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize hierarchical role assignment')
    parser.add_argument('--steps', help='Number of steps to run', type=int, default=500)
    parser.add_argument('--delay', help='Delay between steps (seconds)', type=float, default=0.1)
    parser.add_argument('--verbose', help='Show detailed game state', action='store_true')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    
    # Environment configuration
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 10  # Faster simulation
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    config_dict['team_size'] = 3
    
    print("Hierarchical Role Visualization")
    print("=" * 40)
    print("Watch how roles change based on game state!")
    print(f"Running for {args.steps} steps with {args.delay}s delay between steps")
    print()
    
    try:
        # Create environment with roles
        base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict,
            render_mode=None,  # No rendering for console output
            team_size=3
        )
        
        env = wrap_env_with_roles(base_env)
        
        # Reset environment
        obs, _ = env.reset()
        
        print("Environment created! Starting simulation...")
        print()
        
        # Track role changes
        last_roles = {}
        role_change_count = 0
        
        for step in range(args.steps):
            # Get current roles
            current_roles = {}
            for agent_id in env.agents:
                if agent_id in obs:
                    current_roles[agent_id] = extract_role_from_obs(obs[agent_id])
            
            # Check for role changes
            role_changed = False
            for agent_id in current_roles:
                if agent_id in last_roles and last_roles[agent_id] != current_roles[agent_id]:
                    role_changed = True
                    role_change_count += 1
                    old_role = ROLE_NAMES[last_roles[agent_id]]
                    new_role = ROLE_NAMES[current_roles[agent_id]]
                    print(f"🔄 {agent_id} changed role: {old_role} → {new_role}")
            
            # Print roles periodically or when they change
            if step % 50 == 0 or role_changed:
                print_roles(current_roles, step)
                
                if args.verbose:
                    analyze_game_state(env)
            
            # Random actions
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = env.action_space(agent_id).sample()
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Check if episode ended
            if all(terminated.values()) or all(truncated.values()):
                print(f"\n🏁 Episode ended at step {step}!")
                print_roles(current_roles, step)
                analyze_game_state(env)
                
                # Reset for new episode
                obs, _ = env.reset()
                print("\n--- New Episode Started ---")
            
            # Update last roles
            last_roles = current_roles.copy()
            
            # Delay
            time.sleep(args.delay)
        
        print(f"\n✅ Simulation completed!")
        print(f"Total role changes observed: {role_change_count}")
        
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the environment is properly set up.")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
