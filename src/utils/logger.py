# src/utils/logger.py
"""
Custom logging tools
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Experiment logger
    """
    
    def __init__(self, log_dir: str = "outputs/logs", experiment_name: str = "experiment"):
        """
        Initialize logger
        
        Args:
            log_dir: Log directory
            experiment_name: Experiment name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Set log file paths
        self.log_file = self.log_dir / f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        self.trace_file = self.log_dir / f"{experiment_name}_trace_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Configure logger
        self.logger = logging.getLogger(f"experiment_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize trace data
        self.trace_data = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'episodes': []
        }
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
        
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
        
    def log_episode_start(self, episode_id: int, task_description: str):
        """
        Log episode start
        
        Args:
            episode_id: Episode ID
            task_description: Task description
        """
        self.info(f"Episode {episode_id} started: {task_description}")
        
        # Initialize episode trace
        episode_data = {
            'episode_id': episode_id,
            'task_description': task_description,
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        
        self.trace_data['episodes'].append(episode_data)
        
    def log_step(self, episode_id: int, step_data: Dict[str, Any]):
        """
        Log step information
        
        Args:
            episode_id: Episode ID
            step_data: Step data
        """
        # Find corresponding episode
        for episode in self.trace_data['episodes']:
            if episode['episode_id'] == episode_id:
                step_data['timestamp'] = datetime.now().isoformat()
                episode['steps'].append(step_data)
                break
        
        # Log simplified step information to log file
        self.info(f"Episode {episode_id} - Step {step_data.get('step_id', 0)}: {step_data.get('action', 'Unknown action')}")
        
    def log_episode_end(self, episode_id: int, success: bool, total_steps: int, total_reward: float, failure_reason: str = ""):
        """
        Log episode end
        
        Args:
            episode_id: Episode ID
            success: Whether successful
            total_steps: Total steps
            total_reward: Total reward
            failure_reason: Failure reason
        """
        self.info(f"Episode {episode_id} ended - Success: {success}, Steps: {total_steps}, Reward: {total_reward}")
        
        # Update episode data
        for episode in self.trace_data['episodes']:
            if episode['episode_id'] == episode_id:
                episode['end_time'] = datetime.now().isoformat()
                episode['success'] = success
                episode['total_steps'] = total_steps
                episode['total_reward'] = total_reward
                episode['failure_reason'] = failure_reason
                break
    
    def log_skill_extraction(self, episode_id: int, skills_extracted: int):
        """
        Log skill extraction
        
        Args:
            episode_id: Episode ID
            skills_extracted: Number of skills extracted
        """
        self.info(f"Episode {episode_id} - Extracted {skills_extracted} new skills")
        
    def log_skill_retrieval(self, episode_id: int, relevant_skills: int):
        """
        Log skill retrieval
        
        Args:
            episode_id: Episode ID
            relevant_skills: Number of relevant skills
        """
        self.info(f"Episode {episode_id} - Retrieved {relevant_skills} relevant skills")
        
    def save_trace(self):
        """Save trace data to file"""
        try:
            self.trace_data['end_time'] = datetime.now().isoformat()
            
            with open(self.trace_file, 'w', encoding='utf-8') as f:
                json.dump(self.trace_data, f, ensure_ascii=False, indent=2)
                
            self.info(f"Trace data saved to {self.trace_file}")
            
        except Exception as e:
            self.error(f"Failed to save trace data: {e}")
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """
        Get trace summary
        
        Returns:
            Trace summary information
        """
        total_episodes = len(self.trace_data['episodes'])
        successful_episodes = sum(1 for ep in self.trace_data['episodes'] if ep.get('success', False))
        
        return {
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'success_rate': successful_episodes / total_episodes if total_episodes > 0 else 0.0,
            'log_file': str(self.log_file),
            'trace_file': str(self.trace_file)
        }


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup standard logger
    
    Args:
        name: Logger name
        log_level: Log level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
