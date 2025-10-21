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

import argparse
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew

RENDER_MODE = 'human'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 2v2 PyQuaticus environment')
    parser.add_argument('policy_one', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
    parser.add_argument('policy_two', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 

    reward_config = {}
    args = parser.parse_args()
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 8
    config_dict['max_score'] = 100
    config_dict['max_time']=360
    config_dict['tagging_cooldown'] = 55
    config_dict['tag_on_oob']=True

    #Create Environment
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode='human',reward_config=reward_config, team_size=2)

    obs, info = env.reset()

    # Heuristic opponents (use env, not Team as the second arg)
    H_one = BaseDefender('agent_2', env, mode='easy')
    H_two = BaseAttacker('agent_3', env, mode='easy')
    
    import os
    ckpt1 = os.path.abspath(args.policy_one)
    ckpt2 = os.path.abspath(args.policy_two)

    def find_algo_checkpoint(base_path: str) -> str | None:
        """Search recursively under base_path for RLlib checkpoints.
        Prefer directories named checkpoint_*, but accept files named checkpoint_* as well.
        If none found, return a directory containing .rllib_checkpoint.json.
        """
        best = None
        # First pass: directories named checkpoint_*
        for root, dirs, files in os.walk(base_path):
            for d in dirs:
                if d.startswith('checkpoint_'):
                    cand = os.path.join(root, d)
                    if best is None or cand > best:
                        best = cand
        if best is not None:
            return best
        # Second pass: files named checkpoint_*
        for root, dirs, files in os.walk(base_path):
            for f in files:
                if f.startswith('checkpoint_'):
                    return os.path.join(root, f)
        # Third pass: parent dir containing RLlib checkpoint marker
        for root, dirs, files in os.walk(base_path):
            if '.rllib_checkpoint.json' in files:
                return root
        return None

    # Always load Algorithm(s) and fetch policy IDs during action computation
    from ray.rllib.algorithms.ppo import PPO
    def resolve_algo(path: str) -> PPO:
        p = path
        if not (os.path.exists(p) and os.path.basename(p).startswith('checkpoint_')):
            discovered = find_algo_checkpoint(p)
            if discovered is None:
                raise ValueError(f"Could not find a checkpoint under {p}. Pass a valid RLlib checkpoint path or a folder containing one.")
            p = discovered
        return PPO.from_checkpoint(p)

    algo1 = resolve_algo(ckpt1)
    algo2 = resolve_algo(ckpt2) if ckpt2 != ckpt1 else algo1
    step = 0
    max_step = 2500

    while True:
        new_obs = {}
        #Get Unnormalized Observation for heuristic agents (H_one, and H_two)
        for k in obs:
            new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

        #Get learning agent action from policy
        # Compute actions using algorithm(s) and explicit policy IDs
        r0 = algo1.compute_single_action(obs['agent_0'], policy_id="agent-0-policy")
        zero = r0[0] if isinstance(r0, (list, tuple)) else r0
        try:
            r1 = algo2.compute_single_action(obs['agent_1'], policy_id="agent-1-policy")
        except Exception:
            # Fallback: use algo1 if algo2 doesn't have the policy
            r1 = algo1.compute_single_action(obs['agent_1'], policy_id="agent-1-policy")
        one = r1[0] if isinstance(r1, (list, tuple)) else r1
        #Compute Heuristic agent actions
        two = H_one.compute_action(new_obs, info)
        three = H_two.compute_action(new_obs, info)
        
        #Step the environment
        #Opponents are BaseDefender & BaseAttacker
        #obs, reward, term, trunc, info = env.step({'blue_0':zero, 'blue_1':one,'red_0':two,'red_1':three})

        #Opponents Don't Move:
        obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':-1, 'agent_3':-1})
        k =  list(term.keys())
        if step >= max_step:
            break
        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            env.reset()
    env.close()


