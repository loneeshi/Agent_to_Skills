# src/adapters/alfworld_adapter.py
import os
import sys
import yaml

# Add the local alfworld directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'alfworld'))

import alfworld.agents.environment as alf_env
from agentscope.tool import ToolResponse
from .base import BaseEnvAdapter

class AlfworldAdapter(BaseEnvAdapter):
    def __init__(self, config_path=None):
        # 1. Load ALFWorld Config
        if config_path is None:
            # Try to find the default config in common locations
            possible_config_paths = [
                # Local development path
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'alfworld', 'configs', 'base_config.yaml'),
                # System installation path
                '/usr/local/share/alfworld/configs/base_config.yaml',
                # User home path
                os.path.expanduser('~/.alfworld/configs/base_config.yaml'),
                # Current directory
                'base_config.yaml'
            ]
            
            found_config_path = None
            for path in possible_config_paths:
                if os.path.exists(path):
                    found_config_path = path
                    break
            
            if found_config_path is None:
                # Use a minimal default config instead of failing
                print("Warning: ALFWorld config not found, using minimal default config")
                self.config = self._get_default_config()
            else:
                print(f"Using ALFWorld config from: {found_config_path}")
                with open(found_config_path) as reader:
                    self.config = yaml.safe_load(reader)
        else:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"ALFWorld config not found at: {config_path}")
            
            with open(config_path) as reader:
                self.config = yaml.safe_load(reader)

        # 2. Initialize Environment using ALFWorld's environment system
        # 'eval_out_of_distribution' means we test on unseen tasks (standard benchmark setting)
        env_type = self.config.get('env', {}).get('type', 'AlfredTWEnv')
        try:
            # Create the environment manager
            self.env_manager = alf_env.get_environment(env_type)(self.config, train_eval='eval_out_of_distribution')
        except (AttributeError, KeyError) as e:
            # Fallback to AlfredTWEnv if the specified type doesn't exist
            print(f"Warning: Environment type '{env_type}' not found ({e}), falling back to AlfredTWEnv")
            self.env_manager = alf_env.AlfredTWEnv(self.config, train_eval='eval_out_of_distribution')
        
        # Initialize the actual gym environment with batch_size=1
        self.env = self.env_manager.init_env(batch_size=1)
        
        # 3. Initialize internal state
        self.obs = None
        self.info = None
        self.is_done = False
        self.last_info = {}

    def reset(self) -> str:
        # Reset the environment and get initial observation
        # Use the correct ALFWorld API - reset() instead of init_game
        self.obs, self.infos = self.env.reset()
        
        self.is_done = False
        self.last_info = {}
        
        # ALFWorld returns a list of observations (batch). We take the first one.
        # The task description is usually the string returned after reset.
        if isinstance(self.obs, list) and len(self.obs) > 0:
            task_description = self.obs[0].split("\n")[-1]
        else:
            task_description = str(self.obs)
        return task_description

    def step(self, action_command: str) -> ToolResponse:
        """Execute a text command in the household environment.
        
        Args:
            action_command: A specific action command (e.g., 'go to kitchen', 'take apple 1')
            
        Returns:
            ToolResponse with observation text and metadata
        """
        if self.is_done:
            return ToolResponse(
                content="Error: Task finished.",
                metadata={
                    "done": True,
                    "won": False,
                    "reward": 0.0
                }
            )

        # Execute action in the environment
        # ALFWorld expects a list of actions for batch processing
        obs, scores, dones, infos = self.env.step([action_command])
        
        self.obs = obs[0]
        self.is_done = dones[0]
        self.last_info = infos
        
        # 'won' is the authoritative success flag in ALFWorld info dict
        # infos is a dict of lists (e.g. {'won': [True]})
        is_success = infos.get('won', [False])[0]
        
        # Ensure we have a proper reward value
        reward = scores[0] if scores and len(scores) > 0 else 0.0
        
        obs_text = f"Observation: {self.obs}"
        if self.is_done:
            if is_success:
                obs_text += "\n[System]: SUCCESS! Task completed."
            else:
                obs_text += "\n[System]: FAILED. Task ended."
        
        return ToolResponse(
            content=obs_text,
            metadata={
                "done": self.is_done,
                "won": is_success,
                "reward": reward
            }
        )

    def _get_default_config(self):
        """Provide a minimal default config when ALFWorld config is not found"""
        return {
            'dataset': {
                'data_path': 'json_2.1.1/train',
                'eval_id_data_path': 'json_2.1.1/valid_seen',
                'eval_ood_data_path': 'json_2.1.1/valid_unseen',
                'num_train_games': -1,
                'num_eval_games': -1
            },
            'logic': {
                'domain': 'logic/alfred.pddl',
                'grammar': 'logic/alfred.twl2'
            },
            'env': {
                'type': 'AlfredTWEnv',
                'domain_randomization': False,
                'task_types': [1, 2, 3, 4, 5, 6],
                'expert_timeout_steps': 150,
                'expert_type': 'handcoded',
                'goal_desc_human_anns_prob': 0.0
            },
            'controller': {
                'type': 'oracle',
                'debug': False,
                'load_receps': True
            },
            'general': {
                'random_seed': 42,
                'use_cuda': False,
                'task': 'alfred',
                'training_method': 'dagger',
                'observation_pool_capacity': 3,
                'hide_init_receptacles': False
            },
            'dagger': {
                'action_space': 'generation',
                'max_target_length': 20,
                'beam_width': 10,
                'generate_top_k': 5,
                'unstick_by_beam_search': False,
                'training': {
                    'max_nb_steps_per_episode': 50
                },
                'fraction_assist': {
                    'fraction_assist_anneal_episodes': 50000,
                    'fraction_assist_anneal_from': 1.0,
                    'fraction_assist_anneal_to': 0.01
                },
                'fraction_random': {
                    'fraction_random_anneal_episodes': 0,
                    'fraction_random_anneal_from': 0.0,
                    'fraction_random_anneal_to': 0.0
                },
                'replay': {
                    'replay_memory_capacity': 500000,
                    'update_per_k_game_steps': 5,
                    'replay_batch_size': 64,
                    'replay_sample_history_length': 4,
                    'replay_sample_update_from': 2
                }
            }
        }

    def get_tool_config(self) -> dict:
        return {
            "function": self.step,
            "function_name": "interact_with_environment",
            "function_desc": "Execute a text command in the household environment. Input should be a specific action command (e.g., 'go to kitchen', 'take apple 1')."
        }
