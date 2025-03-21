import numpy as np
from typing import Tuple, List, Set
import heapq

class Robot:
    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos
        self.target = None
        self.path = []
        self.has_item = False
        self.idle = True

class WarehouseSimulator:
    def __init__(self, layout: np.ndarray, num_robots: int, num_workstations: int, target_orders: int):
        self.layout = layout
        self.rows, self.cols = layout.shape
        self.num_robots = num_robots
        self.num_workstations = num_workstations
        self.target_orders = target_orders
        
        # Initialize state
        self.robots = []
        self.workstations = []
        self.storage_locations = []
        self.completed_orders = 0
        self.steps = 0
        
        # Find storage locations (shelves)
        for i in range(self.rows):
            for j in range(self.cols):
                if layout[i, j] == 1:  # Storage shelf
                    self.storage_locations.append((i, j))
        
        # Place workstations at the left edge
        ws_spacing = self.rows // (num_workstations + 1)
        for i in range(num_workstations):
            self.workstations.append((ws_spacing * (i + 1), 0))
        
        # Place robots randomly in empty spaces or aisles
        available_spots = []
        for i in range(self.rows):
            for j in range(self.cols):
                if layout[i, j] <= 0:  # Empty space or road
                    available_spots.append((i, j))
        
        if len(available_spots) < num_robots:
            raise ValueError("Not enough space for all robots")
        
        robot_spots = np.random.choice(len(available_spots), num_robots, replace=False)
        for spot_idx in robot_spots:
            self.robots.append(Robot(available_spots[spot_idx]))
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path using A* algorithm"""
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            # Check all adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                # Check bounds and obstacles
                if (0 <= next_pos[0] < self.rows and 
                    0 <= next_pos[1] < self.cols and 
                    self.layout[next_pos] != 1):  # Not a shelf
                    
                    new_cost = cost_so_far[current] + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + self.manhattan_distance(goal, next_pos)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current
        
        # Reconstruct path
        if goal not in came_from:
            return []
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def assign_tasks(self):
        """Assign tasks to idle robots"""
        for robot in self.robots:
            if robot.idle:
                if not robot.has_item:
                    # Assign pickup task
                    target = np.random.choice(len(self.storage_locations))
                    robot.target = self.storage_locations[target]
                else:
                    # Assign delivery task
                    target = np.random.choice(len(self.workstations))
                    robot.target = self.workstations[target]
                
                robot.path = self.find_path(robot.pos, robot.target)
                if robot.path:
                    robot.idle = False
    
    def move_robots(self) -> Set[Tuple[int, int]]:
        """Move robots along their paths, returns set of occupied positions"""
        occupied = set()
        
        # First, reserve next positions
        for robot in self.robots:
            if not robot.idle and len(robot.path) > 1:
                next_pos = robot.path[1]
                if next_pos in occupied:
                    # Position already taken, stay in place
                    robot.path = self.find_path(robot.pos, robot.target)
                else:
                    occupied.add(next_pos)
        
        # Then move robots
        for robot in self.robots:
            if not robot.idle and len(robot.path) > 1:
                next_pos = robot.path[1]
                if next_pos in occupied:
                    robot.path = self.find_path(robot.pos, robot.target)
                else:
                    robot.pos = next_pos
                    robot.path = robot.path[1:]
                    
                    # Check if reached target
                    if len(robot.path) == 1:
                        if robot.has_item:
                            self.completed_orders += 1
                            robot.has_item = False
                        else:
                            robot.has_item = True
                        robot.idle = True
                        robot.target = None
                        robot.path = []
        
        return occupied
    
    def run(self) -> Tuple[int, float]:
        """Run simulation until target orders are completed
        Returns: (completion_time, orders_per_step)
        """
        while self.completed_orders < self.target_orders:
            self.assign_tasks()
            self.move_robots()
            self.steps += 1
        
        return self.steps, self.completed_orders / self.steps 