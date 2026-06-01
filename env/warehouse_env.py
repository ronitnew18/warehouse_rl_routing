import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NoisyWarehouseEnv(gym.Env):
    # Ensure num_obstacles is added here!
    def __init__(self, grid_size=5, noise_level=0.1, num_obstacles=4): 
        super(NoisyWarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.noise_level = noise_level
        self.num_obstacles = num_obstacles # <--- Add this line
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([grid_size, grid_size])
        
        self.goal_pos = np.array([grid_size-1, grid_size-1])
        
        # Initialize the grid and then call the generator
        self.grid = np.zeros((grid_size, grid_size))
        self._generate_obstacles() 
        self.state = np.array([0, 0])

        # Add this method inside your NoisyWarehouseEnv class
    def move_obstacles(self):
        """Randomly shifts one obstacle to a neighboring empty square."""
    # Find all obstacle coordinates
        obs_r, obs_c = np.where(self.grid == 1)
        if len(obs_r) > 0:
           idx = random.randint(0, len(obs_r) - 1)
        r, c = obs_r[idx], obs_c[idx]
        
        # Possible moves for the obstacle
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = random.choice(moves)
        new_r, new_c = r + dr, c + dc
        
        # If new spot is valid, empty, and not Start/Goal, move it
        if (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size and 
            self.grid[new_r, new_c] == 0 and 
            (new_r, new_c) != (0,0) and 
            (new_r, new_c) != (self.grid_size-1, self.grid_size-1)):
            
            self.grid[r, c] = 0 # Old spot empty
            self.grid[new_r, new_c] = 1 # New spot blocked

    def _generate_obstacles(self):
        # Reset the grid to all zeros first
        self.grid = np.zeros((self.grid_size, self.grid_size))
        count = 0
        while count < self.num_obstacles:
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            # Avoid Start (0,0) and Goal
            if (r, c) != (0, 0) and (r, c) != (self.grid_size-1, self.grid_size-1):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 1
                    count += 1

    def get_state_index(state, grid_size):
    # Converts (row, col) -> single integer
    # e.g., (1, 2) in a 5x5 grid becomes 1*5 + 2 = 7
        return state[0] * grid_size + state[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0])
        return self.state, {}

    def step(self, action):

        # 2. Calculate New Position
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

        # 3. Check for Obstacles
        if self.grid[new_pos[0], new_pos[1]] == 1:
            reward = -10  # Collision Penalty
            done = False  # Keep trying
            # State remains same (bumped into wall)
        elif np.array_equal(new_pos, self.goal_pos):
            self.state = new_pos
            reward = 100  # GOAL!
            done = True
        else:
            self.state = new_pos
            reward = -1   # Time/Step Penalty
            done = False

        return self.state, reward, done, False, {}
    
    