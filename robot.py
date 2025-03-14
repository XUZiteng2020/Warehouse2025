import numpy as np
from enum import Enum
import heapq
from typing import List, Tuple, Optional

class RobotStatus(Enum):
    IDLE = "idle"
    WORKING = "working"

class Robot:
    def __init__(self, x: int, y: int):
        self.x = x  # grid coordinates
        self.y = y
        self.status = RobotStatus.IDLE
        self.target_x = None
        self.target_y = None
        self.path = []  # List of coordinates to follow
        self.rotation = 0  # Degrees (0: right, 90: up, 180: left, 270: down)
        self.turning_steps = 0  # Steps remaining in current turn
        self.waiting_time = 0  # Time to wait at current location
        self.is_at_workstation = False

    def get_position_in_front(self) -> Tuple[int, int]:
        """Get the coordinates of the position directly in front of the robot based on its rotation"""
        dx, dy = 0, 0
        if self.rotation == 0:    # Right
            dx, dy = 0, 1
        elif self.rotation == 90:  # Down
            dx, dy = 1, 0
        elif self.rotation == 180:  # Left
            dx, dy = 0, -1
        elif self.rotation == 270:  # Up
            dx, dy = -1, 0
        
        return (self.x + dx, self.y + dy)
    
    def is_robot_in_front(self, all_robots: List['Robot']) -> bool:
        """Check if there's another robot directly in front of this robot"""
        front_pos = self.get_position_in_front()
        
        for other_robot in all_robots:
            # Skip self
            if other_robot is self:
                continue
                
            # Check if any other robot is at the position in front
            if (other_robot.x, other_robot.y) == front_pos:
                print(f"Robot at ({self.x}, {self.y}) detected robot in front at {front_pos}")
                return True
                
        return False

    def update_position(self, warehouse: np.ndarray, all_robots: List['Robot'] = None) -> bool:
        """Update robot position based on current path and turning requirements"""
        if self.waiting_time > 0:
            print(f"Robot at ({self.x}, {self.y}) waiting: {self.waiting_time} steps left")
            self.waiting_time -= 1
            return False

        if not self.path:
            print(f"Robot at ({self.x}, {self.y}) has no path")
            return False

        # Remove any points in path that match current position
        while self.path and self.path[0] == (self.x, self.y):
            self.path.pop(0)
            print(f"Skipping current position in path")
            
        if not self.path:
            return False

        if self.turning_steps > 0:
            print(f"Robot at ({self.x}, {self.y}) turning: {self.turning_steps} steps left")
            self.turning_steps -= 1
            return False

        next_x, next_y = self.path[0]
        print(f"Robot at ({self.x}, {self.y}) moving to ({next_x}, {next_y})")
        
        # Calculate required rotation
        dx = next_x - self.x
        dy = next_y - self.y
        
        # Update target rotation based on movement direction
        target_rotation = None
        if dx > 0:
            target_rotation = 0  # Moving right
        elif dx < 0:
            target_rotation = 180  # Moving left
        elif dy > 0:
            target_rotation = 90  # Moving down
        elif dy < 0:
            target_rotation = 270  # Moving up

        if target_rotation is None:
            print(f"Error: Invalid movement direction dx={dx}, dy={dy}")
            # Clear invalid path point
            if self.path:
                self.path.pop(0)
            return False

        # Check if we need to turn
        if self.rotation != target_rotation:
            # Calculate shortest turning direction
            diff = (target_rotation - self.rotation) % 360
            if diff > 180:
                diff -= 360
            self.turning_steps = abs(diff) // 90 * 5
            print(f"Robot at ({self.x}, {self.y}) starting turn from {self.rotation}° to {target_rotation}°")
            self.rotation = target_rotation
            return False

        # Check for robot directly in front after setting rotation
        if all_robots and self.is_robot_in_front(all_robots):
            print(f"Robot at ({self.x}, {self.y}) waiting for 1 time unit due to robot in front")
            self.waiting_time = 1
            return False

        # Move to next position if no turning needed and no robot in front
        old_x, old_y = self.x, self.y
        self.x, self.y = self.path.pop(0)
        print(f"Robot moved from ({old_x}, {old_y}) to ({self.x}, {self.y})")
        return True

def find_shortest_path(warehouse: np.ndarray, start: Tuple[int, int], 
                      end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A* pathfinding algorithm for robot routing"""
    if start == end:
        return []

    rows, cols = warehouse.shape
    queue = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Find the closest aisle point to the end point if end is on a shelf
    target_points = [end]
    if warehouse[end[0], end[1]] == 1:  # If target is on a shelf
        # Check all adjacent cells for an aisle
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_x, adj_y = end[0] + dx, end[1] + dy
            if (0 <= adj_x < rows and 0 <= adj_y < cols and 
                warehouse[adj_x, adj_y] == 0):
                target_points.append((adj_x, adj_y))
    
    while queue:
        _, current = heapq.heappop(queue)
        
        if current in target_points:
            # If we found an adjacent aisle, add the final step to the shelf
            if current != end:
                came_from[end] = current
            break
            
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = current[0] + dx, current[1] + dy
            
            # Allow movement through aisles and to the final shelf position
            if (0 <= next_x < rows and 0 <= next_y < cols and 
                (warehouse[next_x, next_y] == 0 or (next_x, next_y) == end)):
                
                new_cost = cost_so_far[current] + 1
                next_pos = (next_x, next_y)
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(end, next_pos)
                    heapq.heappush(queue, (priority, next_pos))
                    came_from[next_pos] = current
    
    if end not in came_from:
        print(f"No path found from {start} to {end}")
        return []
        
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    # Remove any duplicate consecutive positions
    if path:
        filtered_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i-1]:
                filtered_path.append(path[i])
        path = filtered_path
    
    print(f"Found path from {start} to {end}: {path}")
    return path 