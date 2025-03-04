import simpy
import numpy as np
import random

class ResourceEnv:
    """
    Enhanced SimPy-based environment for simulating resource allocation tasks.
    
    Adds more realistic task scheduling, resource constraints, and reward mechanisms.
    """

    def __init__(self, state_space=6, resource_capacity=2, max_time=30):
        """
        Initialize the simulation environment with enhanced parameters.
        
        Args:
            state_space: Dimension of state vector
            resource_capacity: Number of resources available
            max_time: Maximum simulation time
        """
        print("Creating enhanced simulation environment...")
        self.env = simpy.Environment()
        self.resource = simpy.Resource(self.env, capacity=resource_capacity)
        self.tasks = []
        self.state_space = state_space
        self.resource_capacity = resource_capacity
        self.simulation_time = 0
        self.max_time = max_time
        self.task_arrival_rate = 0.3  # Probability of new task arrival
        self.waiting_penalty = 0.1    # Penalty for tasks waiting
        self.completion_reward = 1.0  # Reward for task completion
        self.utilization_reward = 0.2 # Reward for resource utilization
        
        # Task queue statistics
        self.completed_tasks = []
        self.waiting_tasks = []
        self.active_tasks = []
        
        # Metrics tracking
        self.avg_waiting_time = 0
        self.resource_utilization = 0
        self.throughput = 0

    def _generate_task(self):
        """
        Generate a new task with realistic properties.
        
        Returns:
            task: Dictionary with task properties
        """
        task_id = len(self.tasks) + 1
        duration = random.randint(2, 8)  # More variable durations
        priority = random.randint(1, 10)
        deadline = self.simulation_time + duration + random.randint(5, 15)
        
        return {
            'task_id': task_id,
            'duration': duration,
            'priority': priority,
            'deadline': deadline,
            'arrival_time': self.simulation_time,
            'start_time': None,
            'completion_time': None,
            'completed': False,
            'expired': False,
            'in_progress': False
        }

    def _task_process(self, task):
        """
        SimPy process for task execution.
        
        Args:
            task: Task to execute
            
        Yields:
            SimPy events
        """
        # Request a resource
        with self.resource.request() as request:
            # Wait for the resource to become available
            yield request
            
            # Mark task as started
            task['in_progress'] = True
            task['start_time'] = self.env.now
            self.active_tasks.append(task)
            if task in self.waiting_tasks:
                self.waiting_tasks.remove(task)
            
            # Execute the task
            yield self.env.timeout(task['duration'])
            
            # Mark task as completed
            task['completed'] = True
            task['in_progress'] = False
            task['completion_time'] = self.env.now
            self.completed_tasks.append(task)
            if task in self.active_tasks:
                self.active_tasks.remove(task)

    def _check_expired_tasks(self):
        """Check for expired tasks (passed deadline)"""
        for task in self.tasks:
            if not task['completed'] and not task['expired'] and self.simulation_time > task['deadline']:
                task['expired'] = True

    def _calculate_metrics(self):
        """Calculate performance metrics"""
        # Average waiting time
        waiting_times = [t['start_time'] - t['arrival_time'] for t in self.completed_tasks 
                         if t['start_time'] is not None]
        self.avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
        
        # Resource utilization (% of time resources were busy)
        self.resource_utilization = self.resource.count / self.resource_capacity
        
        # Throughput (tasks completed per time unit)
        self.throughput = len(self.completed_tasks) / max(1, self.simulation_time)

    def step(self, action):
        """
        Execute a step in the environment based on the action.
        
        Args:
            action: Action to take (0=wait, 1=schedule)
            
        Returns:
            state: New state
            reward: Reward
            done: Whether episode is done
        """
        # print(f"Executing step with action: {action}")
        
        # Process action
        reward = 0
        
        # Action = 1: Schedule a high priority task if available
        if action == 1:
            # Generate a new task with some probability
            if random.random() < self.task_arrival_rate or len(self.waiting_tasks) == 0:
                new_task = self._generate_task()
                self.tasks.append(new_task)
                self.waiting_tasks.append(new_task)
            
            # Schedule the highest priority waiting task
            if self.waiting_tasks:
                # Sort by priority and deadline
                self.waiting_tasks.sort(key=lambda x: (-x['priority'], x['deadline']))
                task_to_schedule = self.waiting_tasks[0]
                
                # Start the task process
                self.env.process(self._task_process(task_to_schedule))
                
                # Small reward for scheduling based on priority
                reward += 0.1 * task_to_schedule['priority'] / 10
        
        # Action = 0: Wait and let current tasks progress
        else:
            # Small penalty for waiting if there are waiting tasks
            reward -= self.waiting_penalty * len(self.waiting_tasks)
        
        # Always add a tick process to ensure the simulation can advance
        self.env.process(self._tick())
        
        # Run the simulation for one time unit
        self.env.run(until=self.env.now + 1)
        self.simulation_time += 1
        
        # Check for expired tasks
        self._check_expired_tasks()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Calculate rewards based on:
        # 1. Task completions
        new_completions = [t for t in self.tasks 
                         if t['completed'] and t['completion_time'] == self.simulation_time]
        reward += self.completion_reward * len(new_completions)
        
        # 2. Resource utilization
        reward += self.utilization_reward * self.resource_utilization
        
        # 3. Penalty for expired tasks
        new_expirations = [t for t in self.tasks 
                         if t['expired'] and not t['completed'] 
                         and not t.get('penalized', False)]
        
        for task in new_expirations:
            reward -= 0.5 * task['priority'] / 10
            task['penalized'] = True
        
        # Build an enhanced state vector with more information:
        # [waiting_tasks, active_tasks, completed_tasks, resource_utilization, 
        #  avg_waiting_time, normalized_time]
        state = np.array([
            len(self.waiting_tasks) / 10.0,  # Normalize number of waiting tasks
            len(self.active_tasks) / self.resource_capacity,  # Normalize active tasks
            len(self.completed_tasks) / 20.0,  # Normalize completed tasks
            self.resource_utilization,  # Already normalized
            min(1.0, self.avg_waiting_time / 10.0),  # Normalize waiting time
            self.simulation_time / self.max_time  # Normalize time
        ])
        
        # Check if done
        done = self.simulation_time >= self.max_time
        
        return state, reward, done

    def _tick(self):
        """SimPy process to ensure time passes"""
        yield self.env.timeout(1)

    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            state: Initial state
        """
        # print("Resetting enhanced environment...")
        self.env = simpy.Environment()
        self.resource = simpy.Resource(self.env, capacity=self.resource_capacity)
        self.tasks = []
        self.completed_tasks = []
        self.waiting_tasks = []
        self.active_tasks = []
        self.simulation_time = 0
        self.avg_waiting_time = 0
        self.resource_utilization = 0
        self.throughput = 0
        
        # Initial state vector
        initial_state = np.zeros(self.state_space)
        
        # Add a process to ensure simulation can advance
        self.env.process(self._tick())
        
        return initial_state

def main():
    """Test the enhanced environment"""
    print("Testing Enhanced ResourceEnv...")
    
    env = ResourceEnv(resource_capacity=2)
    state = env.reset()
    print(f"Initial state: {state}")
    
    # Run a few steps with different actions
    rewards = []
    
    for i in range(10):
        # Alternate between waiting and scheduling
        action = i % 2
        state, reward, done = env.step(action)
        rewards.append(reward)
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
        print(f"State: {state}")
        print(f"Waiting tasks: {len(env.waiting_tasks)}, Active tasks: {len(env.active_tasks)}")
        print(f"Completed tasks: {len(env.completed_tasks)}, Utilization: {env.resource_utilization:.2f}")
        print("---")
    
    print(f"Total reward: {sum(rewards):.2f}")
    print("Enhanced environment test completed.")

if __name__ == "__main__":
    main()