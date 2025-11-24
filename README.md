# Agent Experience Generalization Project

## Overview

This project implements an Agent Experience Generalization system using AgentScope framework with ALFWorld benchmark. The system enables ReAct Agents to develop cross-task skill generalization capabilities for ACL/ICLR submission.

## Project Structure

```
project_root/
├── assets/                  # Static resources
│   ├── prompts/             # All prompt templates (not hardcoded in code)
│   │   ├── system_prompts.yaml   # ReAct agent personality
│   │   └── reflection_prompts.yaml # Skill extraction instructions
│   └── skill_db/            # Core asset: Agent's learned skill library (JSON/VectorDB)
│       └── alfworld_skills.json
│
├── configs/                 # Experiment control center: all hyperparameter configurations
│   ├── default.yaml         # Global configuration
│   ├── envs/                # Environment configurations
│   │   ├── alfworld.yaml
│   │   └── lifelong.yaml
│   └── methods/             # Method configurations (convenient for ablation studies)
│       ├── baseline_react.yaml   # Pure ReAct
│       └── ours_skill_agent.yaml # Agent with memory
│
├── data/                    # Datasets (not Git tracked)
│   ├── alfworld_data/       # Downloaded Alfworld data
│   └── raw_trajectories/    # Raw interaction records (for analyzing failure cases)
│
├── outputs/                 # Experiment results: automatically generated experiment records
│   ├── logs/                # Runtime logs (step-by-step logs)
│   └── results/             # Final metric tables (Success Rate, Steps)
│
├── src/                     # Source code: core logic
│   ├── __init__.py
│   ├── adapters/            # Environment adapter layer: seamless benchmark switching
│   │   ├── base.py          # Defines reset/step interfaces
│   │   ├── alfworld.py      # Alfworld implementation
│   │   └── webarena.py      # Future extension
│   │
│   ├── agents/              # Agent layer
│   │   ├── memory_agent.py  # Your core Agent (inherits from AgentScope)
│   │   └── baselines.py     # Comparison agents
│   │
│   ├── memory/              # Memory module: your novelty is all here
│   │   ├── skill_manager.py # Responsible for retrieval (Retrieve) and storage (Add)
│   │   └── extractor.py     # Responsible for reflection (Reflection Logic)
│   │
│   └── utils/               # Utility functions
│       ├── metrics.py       # Success rate calculation
│       └── logger.py        # Custom logger
│
├── scripts/                 # Launch scripts
│   ├── run_experiment.py    # Main entry: run experiments
│   ├── analyze_results.py   # Plotting: draw learning curves
│   └── visualize_trace.py   # Visualization: generate Case Study HTML
│
├── README.md                # Paper homepage introduction
└── requirements.txt         # Dependencies
```

## Core Architecture

### Environment-as-a-Tool Pattern
The system implements a decoupled architecture where environments are exposed as tools that ReAct Agents can call. This allows easy switching between different benchmarks (ALFWorld, WebArena, LifelongAgentBench) without rewriting agent code.

### Skill Memory System
- **Skill Manager**: Handles skill storage and retrieval using TF-IDF vectorization and cosine similarity
- **Skill Extractor**: Uses LLM to extract reusable skills from successful task execution traces
- **Memory Agent**: Enhanced ReAct Agent with skill retrieval and learning capabilities

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download ALFWorld data
alfworld-download
```

## Usage

### Run Baseline Experiment (Pure ReAct)
```bash
python scripts/run_experiment.py --config configs/default.yaml --method baseline_react
```

### Run Skill Agent Experiment (With Memory)
```bash
python scripts/run_experiment.py --config configs/default.yaml --method ours_skill_agent
```

### Analyze Results
```bash
python scripts/analyze_results.py --results_dir outputs/results
```

### Visualize Traces
```bash
python scripts/visualize_trace.py --trace_file outputs/logs/experiment_trace.json
```

## Key Features

1. **Modular Architecture**: Easy to extend with new environments and agents
2. **Skill Learning**: Automatically extracts and stores reusable skills from successful episodes
3. **Comprehensive Logging**: Detailed experiment tracking and visualization
4. **Configurable Experiments**: YAML-based configuration system for easy experimentation
5. **Metrics Tracking**: Built-in success rate, learning curves, and failure analysis

## Configuration

The system uses a hierarchical configuration system:
- `configs/default.yaml`: Global settings
- `configs/envs/`: Environment-specific settings
- `configs/methods/`: Method-specific settings (baseline vs. skill agent)

## API Keys

Set your DashScope API key:
```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

Or create a `.env` file (make sure it's in .gitignore):
```
DASHSCOPE_API_KEY=your_api_key_here
```

## Development

### Adding New Environments
1. Create a new adapter in `src/adapters/` inheriting from `BaseEnvAdapter`
2. Implement required methods: `reset()`, `step()`, `get_tool_config()`
3. Add environment configuration in `configs/envs/`

### Adding New Agents
1. Create agent class in `src/agents/`
2. Inherit from appropriate AgentScope base class
3. Add method configuration in `configs/methods/`

## License

This project is for research purposes. Please cite appropriately if used in academic work.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
