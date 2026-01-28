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
Interactive test script to visualize hierarchical role assignment in action.

This script creates a 3v3 environment where you can control one team
while the AI controls the other, with real-time role display.
"""

import argparse
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE, K_SPACE, K_LEFT, K_UP, K_RIGHT, K_a, K_w, K_d
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK, DEFEND, INTERCEPT, ROLE_NAMES
import logging

class HierarchicalTest:
    def __init__(self, env, debug_roles=True):
        self.env = env
        self.debug_roles = debug_roles
        self.obs, _ = env.reset()
        self.font = pygame.font.Font(None, 24)
        self.role_history = {agent_id: [] for agent_id in env.agents}
        
        # Action mappings
        self.no_op = 16
        self.straight = 4
        self.left = 6
        self.right = 2
        self.straightleft = 5
        self.straightright = 3
        
        self.blue_keys_to_action = {
            0: self.no_op,
            K_UP: self.straight,
            K_LEFT: self.left,
            K_RIGHT: self.right,
            K_UP + K_LEFT: self.straightleft,
            K_UP + K_RIGHT: self.straightright
        }
        
        self.red_keys_to_action = {
            0: self.no_op,
            K_w: self.straight,
            K_a: self.left,
            K_d: self.right,
            K_w + K_a: self.straightleft,
            K_w + K_d: self.straightright
        }
        
        # Get agent IDs
        self.blue_agent_id = None
        self.red_agent_id = None
        
        for agent_id in env.agents:
            if agent_id.startswith('agent_'):
                agent_num = int(agent_id.split('_')[1])
                if agent_num < 3:  # agents 0,1,2 are typically blue team
                    if self.blue_agent_id is None:
                        self.blue_agent_id = agent_id
                else:  # agents 3,4,5 are typically red team
                    if self.red_agent_id is None:
                        self.red_agent_id = agent_id
    
    def extract_role_from_obs(self, observation):
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
    
    def get_current_roles(self):
        """Get current roles for all agents."""
        roles = {}
        for agent_id in self.env.agents:
            if agent_id in self.obs:
                roles[agent_id] = self.extract_role_from_obs(self.obs[agent_id])
            else:
                roles[agent_id] = ATTACK
        return roles
    
    def draw_role_info(self, screen):
        """Draw role information on screen."""
        roles = self.get_current_roles()
        y_offset = 10
        
        # Title
        title_text = self.font.render("Hierarchical Roles Test", True, (255, 255, 255))
        screen.blit(title_text, (10, y_offset))
        y_offset += 30
        
        # Role information for each agent
        for agent_id in sorted(roles.keys()):
            role_id = roles[agent_id]
            role_name = ROLE_NAMES[role_id]
            
            # Determine team color
            if agent_id.startswith('agent_'):
                agent_num = int(agent_id.split('_')[1])
                color = (100, 100, 255) if agent_num < 3 else (255, 100, 100)  # Blue or Red
            else:
                color = (200, 200, 200)
            
            # Agent info
            agent_text = f"{agent_id}: {role_name}"
            text_surface = self.font.render(agent_text, True, color)
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25
            
            # Add to history
            self.role_history[agent_id].append(role_id)
            if len(self.role_history[agent_id]) > 100:
                self.role_history[agent_id].pop(0)
        
        # Instructions
        y_offset += 20
        instructions = [
            "Blue Team: Arrow Keys",
            "Red Team: WASD Keys", 
            "Space: Pause",
            "ESC: Quit"
        ]
        
        for instruction in instructions:
            inst_text = self.font.render(instruction, True, (200, 200, 200))
            screen.blit(inst_text, (10, y_offset))
            y_offset += 25
        
        # Role statistics
        if self.debug_roles:
            y_offset += 20
            stats_title = self.font.render("Role Distribution (last 100 steps):", True, (255, 255, 255))
            screen.blit(stats_title, (10, y_offset))
            y_offset += 25
            
            for agent_id in sorted(self.role_history.keys()):
                if self.role_history[agent_id]:
                    role_counts = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
                    for role in self.role_history[agent_id]:
                        role_counts[role] += 1
                    
                    total = len(self.role_history[agent_id])
                    if total > 0:
                        stats_text = f"{agent_id}: A:{role_counts[ATTACK]}% D:{role_counts[DEFEND]}% I:{role_counts[INTERCEPT]}%"
                        text_surface = self.font.render(stats_text, True, (180, 180, 180))
                        screen.blit(text_surface, (10, y_offset))
                        y_offset += 20
    
    def process_events(self):
        """Process pygame events and return actions."""
        action_dict = {}
        
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return None  # Signal to quit
        
        # Get key states
        is_key_pressed = pygame.key.get_pressed()
        
        # Blue team controls (arrow keys)
        if self.blue_agent_id:
            blue_keys = (K_RIGHT * is_key_pressed[K_RIGHT] + 
                        K_LEFT * is_key_pressed[K_LEFT] * (is_key_pressed[K_LEFT] - is_key_pressed[K_RIGHT]) + 
                        K_UP * is_key_pressed[K_UP])
            blue_action = self.blue_keys_to_action[blue_keys]
            action_dict[self.blue_agent_id] = blue_action
        
        # Red team controls (WASD)
        if self.red_agent_id:
            red_keys = (K_d * is_key_pressed[K_d] + 
                       K_a * is_key_pressed[K_a] * (is_key_pressed[K_a] - is_key_pressed[K_d]) + 
                       K_w * is_key_pressed[K_w])
            red_action = self.red_keys_to_action[red_keys]
            action_dict[self.red_agent_id] = red_action
        
        # Random actions for other agents
        for agent_id in self.env.agents:
            if agent_id not in action_dict:
                action_dict[agent_id] = self.env.action_space(agent_id).sample()
        
        return action_dict
    
    def run(self):
        """Run the interactive test."""
        clock = pygame.time.Clock()
        running = True
        paused = False
        
        print("Hierarchical Roles Test Started!")
        print("Controls:")
        print("  Blue Team: Arrow Keys")
        print("  Red Team: WASD")
        print("  Space: Pause/Resume")
        print("  ESC: Quit")
        print()
        print("Watch how roles change based on game state!")
        
        while running:
            # Process events
            actions = self.process_events()
            if actions is None:
                running = False
                break
            
            # Handle pause
            keys = pygame.key.get_pressed()
            if keys[K_SPACE]:
                paused = not paused
                time.sleep(0.2)  # Prevent rapid pause/unpause
            
            if not paused:
                # Step environment
                self.obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                # Check if episode ended
                if all(terminated.values()) or all(truncated.values()):
                    print("Episode ended! Resetting...")
                    self.obs, _ = self.env.reset()
                    self.role_history = {agent_id: [] for agent_id in self.env.agents}
            
            # Render
            try:
                self.env.render()
                
                # Draw role information on top
                screen = pygame.display.get_surface()
                if screen:
                    self.draw_role_info(screen)
                    pygame.display.flip()
                
            except Exception as e:
                print(f"Rendering error: {e}")
            
            clock.tick(30)  # 30 FPS
        
        try:
            self.env.close()
        except Exception:
            pass
        print("Test completed!")


def main():
    parser = argparse.ArgumentParser(description='Test hierarchical role assignment interactively')
    parser.add_argument('--debug-roles', help='Show role statistics', action='store_true', default=True)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    
    # Environment configuration
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    config_dict['team_size'] = 3
    config_dict['render_agent_ids'] = True
    config_dict['render_lidar_mode'] = "off"  # Turn off lidar for cleaner view
    
    # Create environment with roles
    try:
        base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict,
            render_mode='human',
            team_size=3
        )
        
        # Wrap with role functionality
        env = wrap_env_with_roles(base_env)
        
        print("Environment created successfully!")
        print("Starting interactive test...")
        
        # Run test
        test = HierarchicalTest(env, debug_roles=args.debug_roles)
        test.run()
        
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure you have pygame installed and the environment is properly set up.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
