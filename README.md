# ResourceRL

A reinforcement learning framework for optimizing resource allocation in computing environments.

## Overview

ResourceRL implements a Proximal Policy Optimization (PPO) reinforcement learning agent that learns to efficiently allocate limited computing resources to competing tasks. The system simulates realistic resource allocation scenarios using SimPy and demonstrates how RL can outperform traditional scheduling algorithms.

## Features

- **Resource Environment Simulation**: SimPy-based environment modeling task scheduling with realistic constraints
- **Advanced RL Agent**: Implements PPO (Proximal Policy Optimization) with separate policy and value networks
- **Multiple Scheduling Strategies**: 
  - Reinforcement Learning (PPO)
  - First-Come-First-Served (FCFS)
  - Shortest Job First (SJF)
- **Performance Benchmarking**: Compare RL performance against traditional scheduling algorithms
- **Detailed Metrics**: Track resource utilization, completion rates, and scheduling efficiency

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TiborThompson/resourcerl.git
   cd resourcerl
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch simpy plotly dash numpy
   ```
   
   Alternatively, use the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## Usage

### Training the RL Agent

Run the main script to train and evaluate the RL agent:

```bash
python main.py
```

Optional arguments:
- `--epochs <number>`: Number of training epochs (default: 100)
- `--resources <number>`: Number of resources in the environment (default: 2) 
- `--seed <number>`: Random seed for reproducibility (default: 42)
- `--no-plotly`: Disable Plotly visualization and only use terminal output

### Benchmarking

Compare the RL agent against traditional scheduling algorithms:

```bash
python benchmarking.py
```

This will:
1. Train a new RL agent
2. Evaluate it against FCFS and SJF schedulers
3. Display comparative performance metrics

## Results

The RL agent consistently outperforms traditional scheduling strategies:

- **FCFS Scheduler**: Average reward ~5.5
- **SJF Scheduler**: Average reward ~5.0  
- **RL Agent**: Average reward ~7.4

This represents a 35-49% improvement over baseline algorithms.

## Project Structure

- `environment.py`: SimPy-based resource allocation environment
- `rl_agent.py`: PPO reinforcement learning agent implementation
- `main.py`: Training and evaluation orchestration
- `benchmarking.py`: Comparative analysis against traditional schedulers
- `visualization.py`: Data visualization utilities

## Requirements

- Python 3.8+
- PyTorch
- SimPy
- NumPy
- Plotly (optional for visualization)
- Dash (optional for interactive dashboards)

## License

MIT

## Acknowledgments

This project draws inspiration from research in reinforcement learning for resource allocation and scheduling optimization.
