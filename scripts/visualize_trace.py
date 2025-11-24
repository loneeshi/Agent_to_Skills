# scripts/visualize_trace.py
"""
Interaction trace visualization script
Used for generating Case Study HTML files
"""

import json
import html
from pathlib import Path
import argparse
from datetime import datetime

def create_html_trace(trace_data, output_path):
    """Create HTML format interaction trace"""
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Interaction Trace - {title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .episode-info {{
            background: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .interaction {{
            margin: 20px 0;
            padding: 15px;
            border-left: 4px solid #007acc;
            background: #f8f9fa;
        }}
        .thought {{
            background: #fff3cd;
            border-left-color: #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        .action {{
            background: #d1ecf1;
            border-left-color: #17a2b8;
            padding: 10px;
            margin: 10px 0;
        }}
        .observation {{
            background: #d4edda;
            border-left-color: #28a745;
            padding: 10px;
            margin: 10px 0;
        }}
        .success {{
            background: #d4edda;
            border-left-color: #28a745;
            font-weight: bold;
        }}
        .failure {{
            background: #f8d7da;
            border-left-color: #dc3545;
            font-weight: bold;
        }}
        .step-number {{
            color: #007acc;
            font-weight: bold;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Agent Interaction Trace</h1>
            <p><strong>Task:</strong> {task_description}</p>
            <p><strong>Episode:</strong> {episode_id} | <strong>Result:</strong> {result_status}</p>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>
        
        <div class="episode-info">
            <h3>📋 Episode Summary</h3>
            <p><strong>Total Steps:</strong> {total_steps}</p>
            <p><strong>Success Rate:</strong> {success_rate}</p>
            <p><strong>Task Type:</strong> {task_type}</p>
        </div>
        
        <h3>🔄 Interaction Steps</h3>
        {interactions_html}
        
        <div class="episode-info">
            <h3>📊 Final Result</h3>
            <p><strong>Status:</strong> {final_status}</p>
            <p><strong>Total Reward:</strong> {total_reward}</p>
            <p><strong>Completion Time:</strong> {completion_time}</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Generate interaction steps HTML
    interactions_html = ""
    for i, step in enumerate(trace_data.get('steps', [])):
        interactions_html += f"""
        <div class="interaction">
            <h4><span class="step-number">Step {i + 1}</span></h4>
            <div class="thought">
                <strong>💭 Thought:</strong><br>
                {html.escape(step.get('thought', 'No thought recorded'))}
            </div>
            <div class="action">
                <strong>🎯 Action:</strong><br>
                {html.escape(step.get('action', 'No action recorded'))}
            </div>
            <div class="observation">
                <strong>👁️ Observation:</strong><br>
                {html.escape(step.get('observation', 'No observation recorded'))}
            </div>
        </div>
        """
    
    # Fill template
    html_content = html_template.format(
        title=html.escape(trace_data.get('task_description', 'Unknown Task')),
        task_description=html.escape(trace_data.get('task_description', 'Unknown Task')),
        episode_id=trace_data.get('episode_id', 'Unknown'),
        result_status='✅ SUCCESS' if trace_data.get('success', False) else '❌ FAILED',
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_steps=len(trace_data.get('steps', [])),
        success_rate=f"{trace_data.get('success_rate', 0):.2%}",
        task_type=html.escape(trace_data.get('task_type', 'Unknown')),
        interactions_html=interactions_html,
        final_status='✅ Task Completed Successfully' if trace_data.get('success', False) else '❌ Task Failed',
        total_reward=trace_data.get('total_reward', 0),
        completion_time=trace_data.get('completion_time', 'Unknown')
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Visualize agent interaction traces')
    parser.add_argument('--trace_file', type=str, required=True, help='Path to trace JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load trace data
    with open(args.trace_file, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)
    
    # Generate output filename
    episode_id = trace_data.get('episode_id', 'unknown')
    output_path = f"{args.output_dir}/trace_episode_{episode_id}.html"
    
    # Create HTML visualization
    create_html_trace(trace_data, output_path)
    
    print(f"Visualization created: {output_path}")

if __name__ == "__main__":
    main()
