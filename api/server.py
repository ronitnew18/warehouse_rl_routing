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


# ==========================
# Request Model
# ==========================

class SimulationRequest(BaseModel):
    start: list
    goal: list
    obstacles: list


# ==========================
# Load Trained Q Table
# ==========================

with open("trained_q_table.pkl", "rb") as f:
    q_table = pickle.load(f)


# ==========================
# RL Path Function
# ==========================

def run_rl(env):

    state, _ = env.reset()

    path = [list(state)]

    done = False

    steps = 0

    while not done and steps < 500:

        state_idx = env.get_state_index(
            state,
            env.grid_size
        )

        action = int(
            np.argmax(
                q_table[state_idx]
            )
        )

        next_state, reward, done, _, _ = env.step(
            action
        )

        path.append(
            list(next_state)
        )

        state = next_state

        steps += 1

    return path



# ==========================
# Dijkstra
# ==========================

def dijkstra(grid, start, goal):

    from heapq import heappush, heappop

    rows = len(grid)
    cols = len(grid[0])

    pq = []

    heappush(pq, (0, start))

    visited = set()

    parent = {}

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while pq:

        cost, current = heappop(pq)

        if current == goal:

            path = []

            node = goal

            while node != start:

                path.append(list(node))
                node = parent[node]

            path.append(list(start))

            return path[::-1]

        if current in visited:
            continue

        visited.add(current)

        for dr, dc in directions:

            nr = current[0] + dr
            nc = current[1] + dc

            nxt = (nr, nc)

            if (
                0 <= nr < rows
                and
                0 <= nc < cols
                and
                grid[nr][nc] != -1
                and
                nxt not in visited
            ):

                if nxt not in parent:
                    parent[nxt] = current

                heappush(
                    pq,
                    (cost + 1, nxt)
                )

    return []


# ==========================
# Health Check
# ==========================
app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)

@app.get("/")
def root():

    return FileResponse("frontend/index.html")

# ==========================
# Simulation Endpoint
# ==========================

@app.post("/simulate")
def simulate(req: SimulationRequest):

    size = 15

    grid = np.zeros((size, size))

    for r, c in req.obstacles:

        if (
            0 <= r < size
            and
            0 <= c < size
        ):
            grid[r][c] = 1

    env = NoisyWarehouseEnv(
    grid_size=size,
    custom_obstacles=req.obstacles,
    start_pos=req.start,
    goal_pos=req.goal
)

    rl_path = run_rl(env)

    dijkstra_path = dijkstra_warehouse(
    grid,
    req.start,
    req.goal
)

    return {

        "rl_path": rl_path,

        "dijkstra_path": dijkstra_path,

        "rl_steps": len(rl_path),

        "dijkstra_steps": len(dijkstra_path)
    }