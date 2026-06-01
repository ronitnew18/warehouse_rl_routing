import sys
import os

# Adds the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmarks import dijkstra_warehouse
# ... rest of your code
import numpy as np

def run_evaluation(env, agent, num_trials=50):
    rl_success = 0
    d_crash = 0
    
    for _ in range(num_trials):
        state, _ = env.reset()
        # 1. Check if Dijkstra's initial path is blocked by dynamic move
        d_path = dijkstra_warehouse(env.grid, (0,0), (env.grid_size-1, env.grid_size-1))
        
        # 2. Run RL Agent
        done = False
        steps = 0
        while not done and steps < 100:
            env.move_obstacles() # World changes every step for maximum stress
            s_idx = state[0] * env.grid_size + state[1]
            action = np.argmax(agent.q_table[s_idx])
            state, _, done, _, _ = env.step(action)
            
            # Check if Dijkstra path is now blocked
            for r, c in d_path:
                if env.grid[r, c] == 1:
                    d_crash += 1
                    break
            steps += 1
            if done: rl_success += 1
            
    print(f"RL Success Rate: {(rl_success/num_trials)*100}%")
    print(f"Dijkstra Failure Rate (Dynamic): {(d_crash/num_trials)*100}%")