# src/utils/metrics.py
"""
Evaluation metrics calculation tools
"""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict


class MetricsCalculator:
    """
    Metrics calculator for computing various evaluation metrics
    """
    
    def __init__(self):
        self.episode_results = []
        self.success_rates = []
        self.step_counts = []
        self.reward_history = []
        
    def add_episode_result(
        self,
        episode_id: int,
        success: bool,
        steps: int,
        total_reward: float,
        task_type: str = "",
        failure_reason: str = ""
    ):
        """
        Add episode result
        
        Args:
            episode_id: Episode ID
            success: Whether successful
            steps: Number of steps
            total_reward: Total reward
            task_type: Task type
            failure_reason: Failure reason
        """
        result = {
            'episode_id': episode_id,
            'success': success,
            'steps': steps,
            'total_reward': total_reward,
            'task_type': task_type,
            'failure_reason': failure_reason
        }
        
        self.episode_results.append(result)
        self.success_rates.append(1.0 if success else 0.0)
        self.step_counts.append(steps)
        self.reward_history.append(total_reward)
    
    def calculate_success_rate(self, window_size: int = None) -> float:
        """
        Calculate success rate
        
        Args:
            window_size: Window size, if None calculate overall success rate
            
        Returns:
            Success rate
        """
        if not self.success_rates:
            return 0.0
            
        if window_size is None:
            return np.mean(self.success_rates)
        else:
            recent_episodes = self.success_rates[-window_size:]
            return np.mean(recent_episodes) if recent_episodes else 0.0
    
    def calculate_average_steps(self, successful_only: bool = False) -> float:
        """
        Calculate average steps
        
        Args:
            successful_only: Whether to only calculate for successful episodes
            
        Returns:
            Average steps
        """
        if not self.step_counts:
            return 0.0
            
        if successful_only:
            successful_steps = [
                steps for result, steps in zip(self.episode_results, self.step_counts)
                if result['success']
            ]
            return np.mean(successful_steps) if successful_steps else 0.0
        else:
            return np.mean(self.step_counts)
    
    def calculate_average_reward(self) -> float:
        """
        Calculate average reward
        
        Returns:
            Average reward
        """
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)
    
    def get_task_type_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get task type statistics
        
        Returns:
            Task type statistics information
        """
        task_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'steps': []})
        
        for result in self.episode_results:
            task_type = result.get('task_type', 'unknown')
            task_stats[task_type]['total'] += 1
            if result['success']:
                task_stats[task_type]['success'] += 1
            task_stats[task_type]['steps'].append(result['steps'])
        
        # Calculate success rate
        for task_type, stats in task_stats.items():
            stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
            stats['avg_steps'] = np.mean(stats['steps']) if stats['steps'] else 0.0
        
        return dict(task_stats)
    
    def get_failure_reason_statistics(self) -> Dict[str, int]:
        """
        Get failure reason statistics
        
        Returns:
            Failure reason statistics
        """
        failure_reasons = defaultdict(int)
        
        for result in self.episode_results:
            if not result['success'] and result.get('failure_reason'):
                failure_reasons[result['failure_reason']] += 1
        
        return dict(failure_reasons)
    
    def get_learning_curve(self, window_size: int = 10) -> List[float]:
        """
        Get learning curve
        
        Args:
            window_size: Sliding window size
            
        Returns:
            Learning curve data
        """
        if not self.success_rates:
            return []
        
        learning_curve = []
        for i in range(len(self.success_rates)):
            start_idx = max(0, i - window_size + 1)
            window = self.success_rates[start_idx:i + 1]
            learning_curve.append(np.mean(window))
        
        return learning_curve
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics
        
        Returns:
            Summary statistics information
        """
        if not self.episode_results:
            return {
                'total_episodes': 0,
                'overall_success_rate': 0.0,
                'average_steps': 0.0,
                'average_reward': 0.0,
                'task_type_stats': {},
                'failure_reasons': {}
            }
        
        return {
            'total_episodes': len(self.episode_results),
            'overall_success_rate': self.calculate_success_rate(),
            'average_steps': self.calculate_average_steps(),
            'average_steps_successful': self.calculate_average_steps(successful_only=True),
            'average_reward': self.calculate_average_reward(),
            'task_type_statistics': self.get_task_type_statistics(),
            'failure_reason_statistics': self.get_failure_reason_statistics(),
            'learning_curve': self.get_learning_curve()
        }
    
    def reset(self):
        """Reset all data"""
        self.episode_results.clear()
        self.success_rates.clear()
        self.step_counts.clear()
        self.reward_history.clear()


def calculate_skill_effectiveness(
    baseline_success_rate: float,
    with_skill_success_rate: float
) -> float:
    """
    Calculate skill effectiveness
    
    Args:
        baseline_success_rate: Baseline success rate
        with_skill_success_rate: Success rate with skills
        
    Returns:
        Skill effectiveness score
    """
    if baseline_success_rate <= 0:
        return 0.0
    
    improvement = with_skill_success_rate - baseline_success_rate
    relative_improvement = improvement / baseline_success_rate
    
    return max(0.0, min(1.0, relative_improvement))


def calculate_skill_usage_impact(
    skill_usage_count: int,
    total_episodes: int,
    skill_success_contribution: float
) -> float:
    """
    Calculate skill usage impact
    
    Args:
        skill_usage_count: Skill usage count
        total_episodes: Total episode count
        skill_success_contribution: Skill contribution to success
        
    Returns:
        Skill usage impact score
    """
    if total_episodes <= 0:
        return 0.0
    
    usage_rate = skill_usage_count / total_episodes
    impact = usage_rate * skill_success_contribution
    
    return max(0.0, min(1.0, impact))
