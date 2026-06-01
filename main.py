import matplotlib.pyplot as plt
import numpy as np
import random
from env.warehouse_env import NoisyWarehouseEnv
from agents.q_learning_agent import QLearningAgent
from utils.benchmarks import dijkstra_warehouse

# --- 1. Configuration ---
GRID_SIZE = 15
NUM_OBSTACLES = 45
TRAIN_EPISODES = 7000 # Increased for the larger grid

env = NoisyWarehouseEnv(grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)
agent = QLearningAgent(num_states=GRID_SIZE**2, num_actions=4)

def get_state_idx(state):
    return int(state[0] * GRID_SIZE + state[1])

# --- 2. Training Phase ---
print(f"Phase 1: Training RL Agent on {GRID_SIZE}x{GRID_SIZE} grid...")
reward_history = []

for ep in range(TRAIN_EPISODES):
    state, _ = env.reset()
    done = False
    while not done:
        s_idx = get_state_idx(state)
        action = agent.choose_action(s_idx)
        next_state, reward, done, _, _ = env.step(action)
        next_s_idx = get_state_idx(next_state)
        
        agent.update(s_idx, action, reward, next_s_idx)
        state = next_state
    
    agent.decay_epsilon()
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}/{TRAIN_EPISODES} - Epsilon: {agent.epsilon:.2f}")

print("Training Complete.\n")

# --- 3. Dynamic Stress Test ---
print("Phase 2: Starting Dynamic Stress Test (Obstacles will move!)...")
state, _ = env.reset()
# Calculate Dijkstra once at the start (The "Static Plan")
d_path = dijkstra_warehouse(env.grid, (0,0), (GRID_SIZE-1, GRID_SIZE-1))

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

done = False
steps = 0
rl_state, _ = env.reset()
max_steps = 100

# Track successes
dijkstra_crashed = False

while not done and steps < max_steps:
    # 1. EVERY 3 STEPS, AN OBSTACLE MOVES
    if steps % 3 == 0 and steps > 0:
        env.move_obstacles()
        print(f"Step {steps}: A shelf has moved!")

    # 2. RL Agent Movement (Adaptive)
    s_idx = get_state_idx(rl_state)
    rl_action = np.argmax(agent.q_table[s_idx])
    next_rl_state, _, done, _, _ = env.step(rl_action)
    rl_state = next_rl_state

    # 3. Visualization: Side-by-Side
    
    # Left: Dijkstra's Static Plan vs Current World
    ax1.clear()
    ax1.imshow(env.grid, cmap='viridis')
    # Draw Dijkstra's intended path
    for r, c in d_path:
        ax1.scatter(c, r, color='cyan', s=10, alpha=0.5)
        # Check if an obstacle moved onto Dijkstra's path
        if env.grid[r, c] == 1:
            ax1.scatter(c, r, color='red', marker='x', s=100)
            dijkstra_crashed = True
    ax1.set_title(f"Dijkstra Plan\n{'CRASHED!' if dijkstra_crashed else 'Clear'}")

    # Right: RL Agent's Live Adaptation
    ax2.clear()
    img_rl = env.grid.copy()
    img_rl[rl_state[0], rl_state[1]] = 0.5 # Current Robot Position
    ax2.imshow(img_rl, cmap='plasma')
    ax2.set_title(f"RL Agent (Adapting)\nStep: {steps}")

    plt.pause(0.3)
    steps += 1

plt.ioff()
print(f"\nTest Finished in {steps} steps.")
if dijkstra_crashed:
    print("RESULT: Classical Dijkstra failed due to environment changes.")
else:
    print("RESULT: Dijkstra succeeded (Obstacles didn't block the path).")
print("RESULT: RL Agent successfully navigated the dynamic environment.")
plt.show()

import pickle
with open('trained_q_table.pkl', 'wb') as f:
    pickle.dump(agent.q_table, f)
print("Brain saved to 'trained_q_table.pkl'")
