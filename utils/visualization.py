import matplotlib.pyplot as plt
import numpy as np

def plot_q_heatmap(agent, grid_size):
    # Calculate Value Function V(s) = max_a Q(s, a)
    v_function = np.max(agent.q_table, axis=1).reshape((grid_size, grid_size))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(v_function, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value (Expected Future Reward)')
    plt.title("Q-Table Value Heatmap (The Agent's Knowledge)")
    plt.show()

import seaborn as sns # You may need to pip install seaborn

def plot_q_values(agent, grid_size):
    # Extract the best value for each state
    v_matrix = np.max(agent.q_table, axis=1).reshape((grid_size, grid_size))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(v_matrix, annot=False, cmap='magma')
    plt.title("The Agent's Knowledge Map (Value Function)")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.show()

import seaborn as sns # You may need to pip install seaborn

def plot_q_values(agent, grid_size):
    # Extract the best value for each state
    v_matrix = np.max(agent.q_table, axis=1).reshape((grid_size, grid_size))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(v_matrix, annot=False, cmap='magma')
    plt.title("The Agent's Knowledge Map (Value Function)")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.show()