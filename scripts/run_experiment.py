import sys
import os
import asyncio
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel, DashScopeChatModel
from agentscope.formatter import DeepSeekChatFormatter

from src.adapters.alfworld_adapter import AlfworldAdapter
from src.agents.memory_agent import MemoryAgent
from src.memory.skill_manager import SkillManager
from src.utils.logger import ExperimentLogger
from src.utils.metrics import MetricsCalculator


class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = ExperimentLogger(
            log_dir=self.config.get('logging', {}).get('log_dir', 'outputs/logs'),
            experiment_name=self.config.get('experiment', {}).get('name', 'agent_experiment')
        )
        self.metrics = MetricsCalculator()
        self.adapter = None
        self.agent = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        method_config_path = Path(f"configs/methods/{config.get('method', 'baseline_react')}.yaml")
        if method_config_path.exists():
            with open(method_config_path, 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)
                config.update(method_config)
                
        return config
    
    def setup_environment(self):
        self.logger.info("Setting up environment...")
        self.adapter = AlfworldAdapter(
            config_path=self.config.get('environment', {}).get('config_path')
        )
        self.logger.info("Environment setup complete")
        
    def setup_agent(self):
        self.logger.info("Setting up agent...")
        
        model_config = self.config.get('model', {})
        model = OpenAIChatModel(
            model_name="DeepSeek-R1",
            # model_name="Qwen3-235B-A22B",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            client_args={
                "base_url": "https://ai.api.coregpu.cn/v1"
            },
            stream=False,
        )
        
        system_prompt = self.config.get('system_prompt', '')
        agent_type = self.config.get('agent', {}).get('type', 'ReActAgent')
        
        if agent_type == 'MemoryAgent':
            skill_manager = SkillManager(
                similarity_threshold=self.config.get('memory', {}).get('skill_similarity_threshold', 0.7),
                max_skills=self.config.get('memory', {}).get('max_skill_library_size', 1000)
            )
            
            self.agent = MemoryAgent(
                name=self.config.get('agent', {}).get('name', 'MemoryAgent'),
                model_config=model_config,
                skill_manager=skill_manager,
                system_prompt=system_prompt,
                verbose=True
            )
        else:
            toolkit = Toolkit()
            tool_cfg = self.adapter.get_tool_config()
            toolkit.register_tool_function(
                tool_cfg["function"],
                func_description=tool_cfg["function_desc"]
            )
            
            self.agent = ReActAgent(
                name=self.config.get('agent', {}).get('name', 'ReActAgent'),
                sys_prompt=system_prompt,
                model=model,
                formatter=DeepSeekChatFormatter(),
                toolkit=toolkit,
                memory=InMemoryMemory(),
                print_hint_msg=True
            )
        
        self.logger.info("Agent setup complete")
        
    async def run_episode(self, episode_id: int) -> Dict[str, Any]:
        self.logger.log_episode_start(episode_id, "ALFWorld Task")
        
        task_description = self.adapter.reset()
        self.logger.info(f"Task: {task_description}")
        
        msg = Msg(name="system", content=f"Your Goal: {task_description}", role="user")
        
        max_steps = self.config.get('experiment', {}).get('max_steps_per_episode', 50)
        success = False
        total_steps = 0
        total_reward = 0.0
        
        try:
            for step in range(max_steps):
                response = await self.agent(msg)
                
                # Get tool result from response metadata
                is_won = False
                step_reward = 0.0
                
                # Check if response has metadata from tool execution
                if hasattr(response, 'metadata') and isinstance(response.metadata, dict):
                    is_won = response.metadata.get('won', False)
                    step_reward = response.metadata.get('reward', 0.0)
                elif hasattr(self.adapter, 'last_info'):
                    info = self.adapter.last_info
                    is_won = info.get('won', [False])[0] if info else False
                    # Also try to get reward from last_info if available
                    if 'reward' in info:
                        step_reward = info['reward'][0] if isinstance(info['reward'], list) else info['reward']
                
                # Additional fallback: check adapter's last observation for reward info
                if step_reward == 0.0 and hasattr(self.adapter, 'obs'):
                    # Try to extract reward from observation if it contains reward info
                    obs_text = str(self.adapter.obs) if self.adapter.obs else ""
                    if "reward:" in obs_text.lower():
                        # Parse reward from observation text if needed
                        pass
                
                step_data = {
                    'step_id': step + 1,
                    'thought': getattr(response, 'thought', ''),
                    'action': getattr(response, 'action', ''),
                    'observation': response.content if hasattr(response, 'content') else '',
                    'reward': step_reward
                }
                self.logger.log_step(episode_id, step_data)
                
                total_reward += step_reward
                
                if is_won:
                    success = True
                    total_steps = step + 1
                    self.logger.info(f"Episode {episode_id}: Task completed successfully!")
                    # Ensure we capture the final reward for successful completion
                    if step_reward == 0.0:
                        # If no reward was captured, assign a positive reward for success
                        total_reward += 1.0
                    break
                
                if hasattr(self.adapter, 'is_done') and self.adapter.is_done:
                    success = False
                    total_steps = step + 1
                    self.logger.info(f"Episode {episode_id}: Task failed.")
                    break
                
                msg = Msg(name="system", content=response.content, role="user")
                
        except Exception as e:
            self.logger.error(f"Episode {episode_id} failed with error: {e}")
            success = False
            total_steps = max_steps
        
        # Final check: if task was successful but reward is still 0, assign a success reward
        if success and total_reward == 0.0:
            total_reward = 1.0
            self.logger.info(f"Episode {episode_id}: Assigning success reward of 1.0")
        
        self.logger.log_episode_end(episode_id, success, total_steps, total_reward)
        
        if hasattr(self.agent, 'extract_skills_from_episode'):
            skills = await self.agent.extract_skills_from_episode(success)
            self.logger.log_skill_extraction(episode_id, len(skills))
        
        if hasattr(self.agent, 'reset_episode'):
            self.agent.reset_episode()
        
        return {
            'episode_id': episode_id,
            'success': success,
            'steps': total_steps,
            'total_reward': total_reward,
            'task_description': task_description
        }
    
    async def run_experiment(self):
        self.logger.info("Starting experiment...")
        
        self.setup_environment()
        self.setup_agent()
        
        num_episodes = self.config.get('experiment', {}).get('episodes', 1)
        
        for episode_id in range(1, num_episodes + 1):
            self.logger.info(f"Running episode {episode_id}/{num_episodes}")
            
            try:
                result = await self.run_episode(episode_id)
                
                self.metrics.add_episode_result(
                    episode_id=episode_id,
                    success=result['success'],
                    steps=result['steps'],
                    total_reward=result['total_reward'],
                    task_type=result['task_description']
                )
                
                if episode_id % self.config.get('logging', {}).get('save_frequency', 10) == 0:
                    self.save_intermediate_results(episode_id)
                    
            except Exception as e:
                self.logger.error(f"Episode {episode_id} failed: {e}")
                continue
        
        self.save_final_results()
        self.logger.save_trace()
        self.logger.info("Experiment completed!")
        
    def save_intermediate_results(self, episode_id: int):
        results = {
            'episode_id': episode_id,
            'metrics': self.metrics.get_summary(),
            'config': self.config
        }
        
        output_path = Path("outputs/results") / f"intermediate_results_episode_{episode_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved intermediate results to {output_path}")
        
    def save_final_results(self):
        results = {
            'final_metrics': self.metrics.get_summary(),
            'config': self.config,
            'skill_statistics': getattr(self.agent, 'get_skill_statistics', lambda: {})() if hasattr(self.agent, 'get_skill_statistics') else {}
        }
        
        output_path = Path("outputs/results") / "final_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved final results to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Run Agent Experience Generalization Experiment')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Configuration file path')
    parser.add_argument('--method', type=str, help='Method to use (baseline_react or ours_skill_agent)')
    
    args = parser.parse_args()
    
    try:
        runner = ExperimentRunner(args.config)
        
        if args.method:
            runner.config['method'] = args.method
            
        await runner.run_experiment()
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
