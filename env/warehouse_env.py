import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NoisyWarehouseEnv(gym.Env):
    def __init__(self, grid_size=15, noise_level=0.1, num_obstacles=4, custom_obstacles=None, start_pos=None, goal_pos=None): 
        super(NoisyWarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.noise_level = noise_level
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([grid_size, grid_size])
        
        # 1. Dynamically read the start and goal positions from the frontend layout
        self.start_pos = np.array(start_pos) if start_pos is not None else np.array([0, 0])
        self.goal_pos = np.array(goal_pos) if goal_pos is not None else np.array([grid_size-1, grid_size-1])
        
        self.grid = np.zeros((grid_size, grid_size))
        
        # 2. Assign obstacles sent by the web UI, or generate random ones if running main.py locally
        if custom_obstacles is not None:
            self.num_obstacles = len(custom_obstacles)
            for r, c in custom_obstacles:
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    self.grid[r, c] = 1
        else:
            self.num_obstacles = num_obstacles
            self._generate_obstacles() 
            
        self.state = self.start_pos.copy()

    def move_obstacles(self):
        """Randomly shifts one obstacle to a neighboring empty square."""
        obs_r, obs_c = np.where(self.grid == 1)
        if len(obs_r) > 0:
            idx = random.randint(0, len(obs_r) - 1)
            r, c = obs_r[idx], obs_c[idx]
            
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dr, dc = random.choice(moves)
            new_r, new_c = r + dr, c + dc
            
            if (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size and 
                self.grid[new_r, new_c] == 0 and 
                (new_r, new_c) != tuple(self.start_pos) and 
                (new_r, new_c) != tuple(self.goal_pos)):
                
                self.grid[r, c] = 0 
                self.grid[new_r, new_c] = 1 

    def _generate_obstacles(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        count = 0
        while count < self.num_obstacles:
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if (r, c) != tuple(self.start_pos) and (r, c) != tuple(self.goal_pos):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 1
                    count += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 3. Always clear and return to the selected start position
        self.state = self.start_pos.copy()
        return self.state, {}

    def step(self, action):
        new_pos = self.state.copy()

        if random.random() < self.noise_level:
            action = random.choice([0, 1, 2, 3])

        if action == 0 and self.state[0] > 0: # Up
            new_pos[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1: # Down
            new_pos[0] += 1
        elif action == 2 and self.state[1] > 0: # Left
            new_pos[1] -= 1
        elif action == 3 and self.state[1] < self.grid_size - 1: # Right
            new_pos[1] += 1

        # Check for Obstacles
        if self.grid[new_pos[0], new_pos[1]] == 1:
            reward = -10  # Collision Penalty
            done = False  
        elif np.array_equal(new_pos, self.goal_pos):
            self.state = new_pos
            reward = 100  # GOAL!
            done = True
        else:
            self.state = new_pos
            reward = -1   # Step Penalty
            done = False

        return self.state, reward, done, False, {}
