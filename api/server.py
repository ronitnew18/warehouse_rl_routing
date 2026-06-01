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

def run_rl(env):
    state, _ = env.reset()
    path = [list(state)]
    done = False
    steps = 0

    while not done and steps < 100:
        # Calculate state index directly to avoid class method errors
        state_idx = int(state[0] * env.grid_size + state[1])

        # Prevent index out of bounds if table is smaller than the grid state
        if state_idx >= len(q_table):
            break

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

    # Initialize environment with custom overrides matching your updated env file
    env = NoisyWarehouseEnv(
        grid_size=size,
        custom_obstacles=req.obstacles,
        start_pos=req.start,
        goal_pos=req.goal
    )

    rl_path = run_rl(env)
    dijkstra_path = dijkstra_warehouse(grid, req.start, req.goal)

    return {
        "rl_path": rl_path,
        "dijkstra_path": dijkstra_path,
        "rl_steps": len(rl_path),
        "dijkstra_steps": len(dijkstra_path)
    }
