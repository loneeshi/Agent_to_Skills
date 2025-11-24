# scripts/analyze_results.py
"""
Experiment results analysis script
Used for plotting learning curves and analyzing experiment results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_experiment_results(results_dir):
    """Load experiment results"""
    results = []
    for result_file in Path(results_dir).glob("*.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            results.append(result)
    return results

def plot_learning_curve(results, output_path):
    """Plot learning curve"""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='episode', y='success_rate', marker='o')
    plt.title('Learning Curve: Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def analyze_failure_cases(results, output_path):
    """Analyze failure cases"""
    failure_cases = [r for r in results if not r.get('success', False)]
    
    analysis = {
        'total_episodes': len(results),
        'failure_count': len(failure_cases),
        'failure_rate': len(failure_cases) / len(results) if results else 0,
        'common_failure_reasons': {}
    }
    
    # Count common failure reasons
    for case in failure_cases:
        reason = case.get('failure_reason', 'unknown')
        analysis['common_failure_reasons'][reason] = analysis['common_failure_reasons'].get(reason, 0) + 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, default='outputs/results', help='Results directory')
    parser.add_argument('--output_dir', type=str, default='outputs/analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("No results found to analyze")
        return
    
    # Plot learning curve
    plot_learning_curve(results, f"{args.output_dir}/learning_curve.png")
    
    # Analyze failure cases
    analysis = analyze_failure_cases(results, f"{args.output_dir}/failure_analysis.json")
    
    print(f"Analysis complete:")
    print(f"- Total episodes: {analysis['total_episodes']}")
    print(f"- Failure rate: {analysis['failure_rate']:.2%}")
    print(f"- Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
