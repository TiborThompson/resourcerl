import environment
import rl_agent
import visualization
import argparse
import numpy as np
import torch
import random

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def initialize_environment(resource_capacity=2, max_time=30):
    """
    Initialize the resource allocation environment.
    
    Args:
        resource_capacity: Number of resources available
        max_time: Maximum simulation time
        
    Returns:
        ResourceEnv: An instance of the resource allocation environment
    """
    print("Initializing the resource environment...")
    try:
        env = environment.ResourceEnv(
            resource_capacity=resource_capacity,
            max_time=max_time
        )
        print("Environment initialized successfully.")
        return env
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        raise

def initialize_agent(env, lr=3e-4):
    """
    Initialize the reinforcement learning agent.
    
    Args:
        env: The environment
        lr: Learning rate
        
    Returns:
        AdvancedRLAgent: An instance of the RL agent
    """
    print("Initializing the advanced RL agent...")
    try:
        agent = rl_agent.AdvancedRLAgent(env, lr=lr)
        print("Agent initialized successfully.")
        return agent
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        raise

def train_agent(agent, epochs=100, verbose=True):
    """
    Train the reinforcement learning agent.
    
    Args:
        agent: The agent to train
        epochs: Number of training epochs
        verbose: Whether to print training progress
    """
    print(f"Starting agent training for {epochs} epochs...")
    try:
        agent.train(epochs=epochs, verbose=verbose)
        print("Agent training completed successfully.")
    except Exception as e:
        print(f"Failed during agent training: {e}")
        raise

def evaluate_agent(agent, num_episodes=5):
    """
    Evaluate the trained agent over multiple episodes.
    
    Args:
        agent: The trained agent
        num_episodes: Number of evaluation episodes
        
    Returns:
        list: Evaluation rewards
    """
    print(f"Evaluating agent over {num_episodes} episodes...")
    evaluation_rewards = []
    
    for i in range(num_episodes):
        state = agent.env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Use only the action selection part of the agent
            action, _, _ = agent.select_action(state)
            next_state, reward, done = agent.env.step(action)
            episode_reward += reward
            state = next_state
        
        evaluation_rewards.append(episode_reward)
        print(f"Evaluation episode {i+1}: reward={episode_reward:.2f}")
    
    avg_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    print(f"Average evaluation reward: {avg_reward:.2f}")
    
    return evaluation_rewards

def visualize_results(agent, evaluation_rewards=None, use_plotly=False):
    """
    Display the training and evaluation results in the terminal.
    
    Args:
        agent: The trained agent
        evaluation_rewards: List of evaluation rewards
        use_plotly: Whether to use Plotly visualization (ignored, always uses terminal)
    """
    print("\n===== ResourceRL Training Results =====")
    training_results = agent.results
    
    # Basic statistics
    print(f"Total training episodes: {len(training_results)}")
    print(f"Final reward: {training_results[-1]:.2f}")
    print(f"Max reward: {max(training_results):.2f} (episode {training_results.index(max(training_results))+1})")
    print(f"Min reward: {min(training_results):.2f} (episode {training_results.index(min(training_results))+1})")
    print(f"Average reward: {sum(training_results)/len(training_results):.2f}")
    
    # Calculate trend
    if len(training_results) > 1:
        x = np.array(range(len(training_results)))
        slope, _ = np.polyfit(x, training_results, 1)
        trend_direction = "improving" if slope > 0 else "declining"
        print(f"Training trend: {trend_direction} ({slope:.4f} per episode)")
    
    # Print evaluation results
    if evaluation_rewards:
        print("\n===== Evaluation Results =====")
        print(f"Evaluation episodes: {len(evaluation_rewards)}")
        print(f"Average evaluation reward: {sum(evaluation_rewards)/len(evaluation_rewards):.2f}")
        print(f"Max evaluation reward: {max(evaluation_rewards):.2f}")
        
        # Compare to training
        train_avg = sum(training_results[-len(evaluation_rewards):])/len(evaluation_rewards)
        print(f"Recent training avg: {train_avg:.2f}")
        diff = (sum(evaluation_rewards)/len(evaluation_rewards)) - train_avg
        print(f"Evaluation vs Training: {'+' if diff > 0 else ''}{diff:.2f}")
    
    print("\nResource allocation optimization completed successfully!")

def main():
    """Main function to run the project"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run enhanced ResourceRL")
    parser.add_argument("--no-plotly", action="store_true", help="Disable Plotly visualization")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resources", type=int, default=2, help="Number of resources in the environment")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"Running Enhanced ResourceRL with settings:")
    print(f"  - Training epochs: {args.epochs}")
    print(f"  - Resources: {args.resources}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Random seed: {args.seed}")
    print(f"  - Visualization: {'Terminal only' if args.no_plotly else 'Terminal + Plotly'}")
    
    # Initialize environment and agent
    env = initialize_environment(resource_capacity=args.resources)
    agent = initialize_agent(env, lr=args.lr)
    
    # Train the agent
    train_agent(agent, epochs=args.epochs)
    
    # Evaluate the agent
    evaluation_rewards = evaluate_agent(agent, num_episodes=args.eval_episodes)
    
    # Visualize results
    visualize_results(agent, evaluation_rewards, use_plotly=not args.no_plotly)
    
    print("Enhanced ResourceRL execution completed successfully.")

if __name__ == "__main__":
    main()