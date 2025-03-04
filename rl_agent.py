import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from collections import deque

class AdvancedRLAgent:
    """
    Enhanced Proximal Policy Optimization (PPO) agent with proper clipping,
    advantage estimation, and experience replay.
    
    Attributes:
        env: The simulation environment
        policy_net: Policy network for action selection
        value_net: Value network for state value estimation
        policy_optimizer: Optimizer for the policy network
        value_optimizer: Optimizer for the value network
        memory: Experience replay buffer
        results: List to store episode rewards
    """
    
    def __init__(self, env, lr=3e-4, memory_size=1024, gamma=0.99, lambd=0.95, eps_clip=0.2):
        """
        Initialize the agent with environment and hyperparameters.
        
        Args:
            env: Simulation environment
            lr: Learning rate
            memory_size: Size of experience replay buffer
            gamma: Discount factor
            lambd: GAE parameter
            eps_clip: PPO clipping parameter
        """
        self.env = env
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.memory_size = memory_size
        
        # Initialize neural networks
        state_dim = env.state_space
        action_dim = 2  # Binary action space (0 or 1)
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Initialize experience replay buffer
        self.memory = []
        
        # Store results
        self.results = []
    
    def select_action(self, state):
        """
        Select an action based on current policy and state.
        
        Args:
            state: Current environment state
            
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Estimated state value
        """
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state)
        else:
            state_tensor = state.float()
        
        with torch.no_grad():  # No need to track gradients during action selection
            # Get action probabilities and state value
            action_probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
            
            # Create distribution and sample action
            dist = distributions.Categorical(action_probs)
            action = dist.sample()
            
            # Return action, log probability, and value
            return action.item(), dist.log_prob(action), value
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory"""
        # Convert tensors to detached numpy arrays to prevent memory leaks
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.detach()
        if isinstance(value, torch.Tensor):
            value = value.detach()
            
        self.memory.append((state, action, reward, next_state, done, log_prob, value))
        
        # Keep memory at desired size
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def compute_advantages(self):
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Returns:
            states: List of states
            actions: List of actions
            log_probs: List of log probabilities
            returns: List of returns
            advantages: List of advantages
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Extract experiences from memory
        for exp in self.memory:
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            dones.append(exp[4])
            log_probs.append(exp[5])
            values.append(exp[6])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        values = torch.stack(values).squeeze()
        
        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute returns and advantages
        next_value = 0
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            # Compute TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.lambd * (1 - dones[t]) * gae
            
            # Store advantages and returns
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, returns, advantages
    
    def update_policy(self, epochs=10, batch_size=64):
        """
        Update policy and value networks using PPO algorithm with completely
        separate forward and backward passes.
        
        Args:
            epochs: Number of update epochs
            batch_size: Batch size for training
        """
        # Compute advantages and returns
        states, actions, old_log_probs, returns, advantages = self.compute_advantages()
        
        # Update for multiple epochs
        for _ in range(epochs):
            # Create indices for batching
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Process in batches
            for start in range(0, len(states), batch_size):
                # Get batch indices
                idx = indices[start:start + batch_size]
                
                # Extract batch data
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # POLICY NETWORK UPDATE (completely separate)
                action_probs = self.policy_net(batch_states)
                dist = distributions.Categorical(action_probs)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratios and policy loss
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                # Update policy network (completely separate backward pass)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # VALUE NETWORK UPDATE (completely separate forward pass)
                # Detach to ensure no gradient flow between networks
                value_states = batch_states.clone().detach()
                curr_values = self.value_net(value_states).squeeze()
                value_loss = nn.MSELoss()(curr_values, batch_returns)
                
                # Update value network (completely separate backward pass)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
    
    def train(self, epochs=100, update_freq=20, verbose=False):
        """
        Train the agent for multiple epochs
        
        Args:
            epochs: Number of training epochs
            update_freq: Frequency of policy updates
            verbose: Whether to print training progress
            
        Returns:
            results: List of episode rewards
        """
        print("Starting enhanced PPO training process...")
        
        for epoch in range(epochs):
            # if verbose:
                # print(f"Epoch {epoch+1}/{epochs}: Starting episode...")
            
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            # Clear memory at the start of each episode
            self.memory = []
            
            # Run episode
            while not done:
                # Select action
                action, log_prob, value = self.select_action(state)
                
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done, log_prob, value)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            # Update policy after episode
            self.update_policy()
            
            # Store results
            self.results.append(episode_reward)
            
            # if verbose:
            #     print(f"Epoch {epoch+1} completed with reward: {episode_reward:.2f}")
        
        print("Training completed successfully.")
        return self.results

class PolicyNetwork(nn.Module):
    """
    Neural network for policy function.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Initialize the network layers.
        
        Args:
            input_dim: Dimension of input state
            output_dim: Dimension of action space
            hidden_dim: Dimension of hidden layer
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)

class ValueNetwork(nn.Module):
    """
    Neural network for value function.
    """
    
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initialize the network layers.
        
        Args:
            input_dim: Dimension of input state
            hidden_dim: Dimension of hidden layer
        """
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)