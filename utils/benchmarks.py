# utils/benchmarks.py
import heapq
import numpy as np

def dijkstra_warehouse(grid, start, goal):
    """
    Finds the shortest path in the warehouse grid.
    Returns: list of coordinates [(r1, c1), (r2, c2)...]
    """
    rows, cols = grid.shape
    start = tuple(start)
    goal = tuple(goal)
    
    # Priority Queue for Dijkstra: (cost, current_node)
    pq = [(0, start)]
    distances = {start: 0}
    previous_nodes = {start: None}
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node == goal:
            break
            
        # Check 4 directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current_node[0] + dr, current_node[1] + dc)
            
            # Stay within bounds and avoid obstacles (grid value 1)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 0 or neighbor == goal:
                    new_dist = current_dist + 1 # Each step costs 1
                    
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct the path
    path = []
    curr = goal
    if goal not in previous_nodes: return [] # No path found
    
    while curr:
        path.append(curr)
        curr = previous_nodes[curr]
    
    return path[::-1] # Reverse to get Start -> Goal