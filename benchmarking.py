import numpy as np
import environment
import rl_agent
import visualization
import random
import torch

class HeuristicScheduler:
    """
    Implements a heuristic scheduler to allocate resources among tasks.
    
    Attributes:
        strategy: The strategy to use for scheduling ('fcfs', 'priority', 'sjf')

    Methods:
        schedule(tasks): Allocates resources based on the selected heuristic strategy.
    """
    
    def __init__(self, strategy='fcfs'):
        """
        Initializes the heuristic scheduler.
        
        Args:
            strategy: The scheduling strategy to use
                     'fcfs' - First-Come-First-Served
                     'priority' - Priority-based scheduling
                     'sjf' - Shortest Job First
        """
        self.strategy = strategy
        valid_strategies = ['fcfs', 'priority', 'sjf']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        print(f"Initializing HeuristicScheduler with {strategy.upper()} strategy...")

    def schedule(self, tasks):
        """
        Allocates resources based on the selected heuristic strategy.

        Params:
            tasks: List of tasks where each task is represented by a dictionary 
                   containing 'task_id', 'duration', and optionally 'priority'.

        Returns:
            scheduled_tasks: List of tasks in the order they got scheduled.
        """
        print(f"Scheduling tasks using {self.strategy.upper()} strategy...")
        # Ensure tasks are non-empty and properly formatted
        if not tasks or not all(isinstance(task, dict) and 'task_id' in task and 'duration' in task for task in tasks):
            raise ValueError("Tasks should be a list of dictionaries with 'task_id' and 'duration'.")
        
        if self.strategy == 'fcfs':
            # Sort by task_id assuming it indicates arrival order
            scheduled_tasks = sorted(tasks, key=lambda x: x['task_id'])
        elif self.strategy == 'priority':
            # Sort by priority (higher priority first)
            if not all('priority' in task for task in tasks):
                raise ValueError("Priority strategy requires 'priority' field in tasks")
            scheduled_tasks = sorted(tasks, key=lambda x: x['priority'], reverse=True)
        elif self.strategy == 'sjf':
            # Sort by duration (shortest job first)
            scheduled_tasks = sorted(tasks, key=lambda x: x['duration'])
        
        print(f"Tasks scheduled successfully with {self.strategy.upper()} strategy.")
        return scheduled_tasks

def run_benchmark(env_config, agent_trained=None, num_episodes=10, seed=42):
    """
    Run a benchmark comparing different scheduling approaches.
    
    Args:
        env_config: Configuration for the environment
        agent_trained: Optional trained RL agent
        num_episodes: Number of episodes to evaluate
        seed: Random seed for reproducibility
    
    Returns:
        dict: Results of the benchmark
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Running benchmark for {num_episodes} episodes...")
    
    # Initialize environment
    env = environment.ResourceEnv(**env_config)
    
    # Initialize schedulers
    fcfs_scheduler = HeuristicScheduler(strategy='fcfs')
    sjf_scheduler = HeuristicScheduler(strategy='sjf')
    
    # Results storage
    results = {
        'fcfs': [],
        'sjf': [],
        'rl': []
    }
    
    # Run FCFS scheduler
    print("\nEvaluating FCFS scheduler...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Track the number of times we've scheduled consecutively
        consecutive_schedules = 0
        
        while not done:
            # FCFS strategy schedules tasks as soon as they arrive
            # But we need to avoid scheduling too many times in a row without tasks
            if len(env.waiting_tasks) > 0:
                action = 1  # Schedule
                consecutive_schedules = 0
            else:
                # If we have no waiting tasks but resources are not fully utilized
                # Try to generate new tasks occasionally
                if env.resource_utilization < 0.8 and consecutive_schedules < 3:
                    action = 1
                    consecutive_schedules += 1
                else:
                    action = 0  # Wait
                    consecutive_schedules = 0
            
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        
        results['fcfs'].append(episode_reward)
        print(f"Episode {episode+1}: reward={episode_reward:.2f}")
    
    # Run SJF scheduler
    print("\nEvaluating SJF scheduler...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Use a alternating scheduling pattern
            # Schedule more often when utilization is low
            if env.simulation_time % 2 == 0 and env.resource_utilization < 0.7:
                action = 1
            else:
                action = 0
                
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        
        results['sjf'].append(episode_reward)
        print(f"Episode {episode+1}: reward={episode_reward:.2f}")
    
    # Run RL agent if provided
    if agent_trained:
        print("\nEvaluating RL agent...")
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _, _ = agent_trained.select_action(state)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
            
            results['rl'].append(episode_reward)
            print(f"Episode {episode+1}: reward={episode_reward:.2f}")
    
    # Calculate and print averages
    print("\nBenchmark Results Summary:")
    for strategy, rewards in results.items():
        if rewards:  # Only print if we have results
            avg_reward = sum(rewards) / len(rewards)
            print(f"{strategy.upper()} strategy - Avg reward: {avg_reward:.2f}")
    
    return results

def visualize_benchmark(results):
    """
    Display the benchmark results in the terminal.
    
    Args:
        results: Dictionary with strategy names as keys and lists of rewards as values
    """
    print("\n===== ResourceRL Benchmark Results =====")
    
    # If no results, exit early
    if not any(results.values()):
        print("No results to display.")
        return
    
    # Calculate summary statistics for each strategy
    strategies_data = {}
    for strategy, rewards in results.items():
        if not rewards:
            continue
        
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        
        strategies_data[strategy] = {
            'avg': avg_reward,
            'max': max_reward,
            'min': min_reward,
            'count': len(rewards)
        }
    
    # Find the best performing strategy
    if strategies_data:
        best_strategy = max(strategies_data.items(), key=lambda x: x[1]['avg'])[0]
    else:
        best_strategy = None
    
    # Print results in a table format
    print("\nStrategy Performance:")
    print("-" * 60)
    print(f"{'Strategy':<10} | {'Episodes':<8} | {'Avg Reward':<12} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)
    
    for strategy, data in strategies_data.items():
        strategy_name = strategy.upper()
        if strategy == best_strategy:
            strategy_name = f"{strategy_name} *"
        
        print(f"{strategy_name:<10} | {data['count']:<8} | {data['avg']:<12.2f} | {data['min']:<8.2f} | {data['max']:<8.2f}")
    
    print("-" * 60)
    
    if best_strategy:
        print(f"\n* {best_strategy.upper()} is the best performing strategy")
    
    # Print head-to-head comparison if we have RL results
    if 'rl' in strategies_data and len(strategies_data) > 1:
        rl_avg = strategies_data['rl']['avg']
        
        print("\nRL Agent Performance Comparison:")
        for strategy, data in strategies_data.items():
            if strategy != 'rl':
                diff = rl_avg - data['avg']
                diff_percent = (diff / data['avg']) * 100 if data['avg'] != 0 else float('inf')
                
                print(f"- vs {strategy.upper()}: {'+' if diff > 0 else ''}{diff:.2f} ({'+' if diff > 0 else ''}{diff_percent:.1f}%)")
    
    print("\nBenchmark analysis completed.")

def train_rl_agent(env_config, epochs=50, seed=42):
    """
    Train an RL agent for benchmarking.
    
    Args:
        env_config: Environment configuration
        epochs: Number of training epochs
        seed: Random seed
        
    Returns:
        agent: Trained RL agent
    """
    print("Training a new RL agent for benchmarking...")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize environment and agent
    env = environment.ResourceEnv(**env_config)
    agent = rl_agent.AdvancedRLAgent(env, lr=3e-4)
    
    # Train the agent
    print(f"Training agent for {epochs} epochs...")
    agent.train(epochs=epochs, verbose=True)
    print("Agent training completed.")
    
    return agent

def main():
    """
    Main function to run the benchmarking.
    """
    print("Running ResourceRL benchmarking...")
    try:
        # Environment configuration
        env_config = {
            'resource_capacity': 2,
            'max_time': 30
        }
        
        # Train a new agent for benchmarking
        agent = train_rl_agent(env_config, epochs=50)
        
        # Run benchmark
        results = run_benchmark(env_config, agent, num_episodes=5)
        
        # Visualize results
        visualize_benchmark(results)
        
        print("Benchmarking completed successfully.")
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")

if __name__ == '__main__':
    main()