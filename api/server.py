from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import pickle

from env.warehouse_env import NoisyWarehouseEnv
from utils.benchmarks import dijkstra_warehouse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    start: list
    goal: list
    obstacles: list

# Load Trained Q Table
with open("trained_q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_radar_state(current_pos, goal_pos, grid, grid_size):
    """
    Recreates your original state logic: (dx, dy, wall_up, wall_down, wall_left, wall_right)
    """
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]
    
    actions_move = [
        (-1, 0),  # UP
        (1, 0),   # DOWN
        (0, -1),  # LEFT
        (0, 1),   # RIGHT
    ]
    
    radar = []
    for dr, dc in actions_move:
        check_r = current_pos[0] + dr
        check_c = current_pos[1] + dc
        
        if check_r < 0 or check_r >= grid_size or check_c < 0 or check_c >= grid_size or grid[check_r][check_c] == 1:
            radar.append(1)  # Blocked
        else:
            radar.append(0)  # Clear
            
    return (dx, dy, *radar)

def run_rl(env, grid):
    state, _ = env.reset()
    path = [list(state)]
    done = False
    steps = 0

    while not done and steps < 100:
        # Convert coordinate into the radar tuple your original agent expects
        radar_state = get_radar_state(state, env.goal_pos, grid, env.grid_size)

        # Secure check: if your Q-table uses a dictionary for tuple states, query it directly
        if isinstance(q_table, dict):
            if radar_state in q_table:
                action = int(np.argmax(q_table[radar_state]))
            else:
                action = int(np.random.randint(0, 4)) # Fallback if state unseen
        else:
            # If your Q-table is a flat array, map state or index safely
            state_idx = int(state[0] * env.grid_size + state[1])
            action = int(np.argmax(q_table[state_idx]))

        next_state, reward, done, _, _ = env.step(action)
        path.append(list(next_state))
        state = next_state
        steps += 1

    return path

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post("/simulate")
def simulate(req: SimulationRequest):
    size = 15
    
    grid = np.zeros((size, size))
    for r, c in req.obstacles:
        if 0 <= r < size and 0 <= c < size:
            grid[r][c] = 1

    env = NoisyWarehouseEnv(
        grid_size=size,
        custom_obstacles=req.obstacles,
        start_pos=req.start,
        goal_pos=req.goal
    )

    rl_path_raw = run_rl(env, grid)
    dijkstra_path_raw = dijkstra_warehouse(grid, req.start, req.goal)

    rl_path = [[int(coords[0]), int(coords[1])] for coords in rl_path_raw]
    dijkstra_path = [[int(coords[0]), int(coords[1])] for coords in dijkstra_path_raw]

    return {
        "rl_path": rl_path,
        "dijkstra_path": dijkstra_path,
        "rl_steps": len(rl_path),
        "dijkstra_steps": len(dijkstra_path)
    }
