import numpy as np
from enum import Enum
import heapq
import random
from typing import List, Tuple, Optional

class RobotStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"  # New status for collision avoidance

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
        self.collision_wait = 0  # New counter for collision waiting
        self.total_wait_time = 0  # Track total time spent waiting
        self.deadlock_threshold = 10  # Reduced threshold for faster response
        self.original_path = []  # Store original path for reverting if needed
        self.shelf_wait = 0  # Time to wait at shelf for item retrieval
        self.exiting_shelf = False  # Flag to indicate robot is trying to exit shelf
        self.shelf_entry_point = None  # Store the position from which robot entered shelf
        self.stuck_count = 0  # Counter for consecutive failed movement attempts
        self.last_position = (x, y)  # Track last position to detect being stuck
        self.backup_attempts = 0  # Track number of backup attempts
        self.shelf_entry_rotation = None  # Store entry rotation

    def check_collision(self, next_x: int, next_y: int, all_robots: List['Robot']) -> bool:
        """Check if moving to the next position would cause a collision"""
        # Calculate movement direction
        dx = next_x - self.x
        dy = next_y - self.y
        
        for other_robot in all_robots:
            if other_robot is self:
                continue
            
            # Only check for robots in our movement direction
            # For horizontal movement
            if dx != 0 and dy == 0:
                # Only check robots in the same row
                if other_robot.y == self.y:
                    # Check if robot is blocking our path
                    if dx > 0:  # Moving right
                        if other_robot.x > self.x and other_robot.x <= next_x:
                            print(f"Robot at ({self.x}, {self.y}) detected collision with robot at ({other_robot.x}, {other_robot.y}) while moving right")
                            return True
                    else:  # Moving left
                        if other_robot.x < self.x and other_robot.x >= next_x:
                            print(f"Robot at ({self.x}, {self.y}) detected collision with robot at ({other_robot.x}, {other_robot.y}) while moving left")
                            return True
            
            # For vertical movement
            elif dy != 0 and dx == 0:
                # Only check robots in the same column
                if other_robot.x == self.x:
                    # Check if robot is blocking our path
                    if dy > 0:  # Moving down
                        if other_robot.y > self.y and other_robot.y <= next_y:
                            print(f"Robot at ({self.x}, {self.y}) detected collision with robot at ({other_robot.x}, {other_robot.y}) while moving down")
                            return True
                    else:  # Moving up
                        if other_robot.y < self.y and other_robot.y >= next_y:
                            print(f"Robot at ({self.x}, {self.y}) detected collision with robot at ({other_robot.x}, {other_robot.y}) while moving up")
                            return True
        
        return False

    def try_alternative_path(self, warehouse: np.ndarray, all_robots: List['Robot']) -> bool:
        """Try to find an alternative path to target avoiding other robots"""
        if self.target_x is None or self.target_y is None:
            return False
            
        # Create a temporary obstacle map adding other robots' positions
        temp_warehouse = warehouse.copy()
        for robot in all_robots:
            if robot is not self:
                temp_warehouse[robot.x, robot.y] = 1  # Mark as obstacle
                
        # Try to find a new path
        new_path = find_shortest_path(temp_warehouse, (self.x, self.y), (self.target_x, self.target_y))
        if new_path:
            print(f"Robot at ({self.x}, {self.y}) found alternative path to ({self.target_x}, {self.target_y})")
            self.path = new_path
            self.total_wait_time = 0
            return True
        return False

    def can_back_off(self, warehouse: np.ndarray, all_robots: List['Robot']) -> Optional[Tuple[int, int]]:
        """Check if robot can move to a nearby empty cell to clear deadlock"""
        # Only check aisle positions (where warehouse value is 0)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = self.x + dx, self.y + dy
            # Check if position is a valid aisle cell
            if (0 <= new_x < warehouse.shape[0] and 
                0 <= new_y < warehouse.shape[1] and 
                warehouse[new_x, new_y] == 0):  # Must be an aisle
                
                # Check if position is free of other robots
                position_free = True
                for robot in all_robots:
                    if robot is not self and robot.x == new_x and robot.y == new_y:
                        position_free = False
                        break
                
                if position_free:
                    # Verify we can find a path back to our target from this position
                    if self.target_x is not None and self.target_y is not None:
                        test_path = find_shortest_path(warehouse, (new_x, new_y), 
                                                     (self.target_x, self.target_y))
                        if not test_path:  # If no path found, this position is not useful
                            continue
                    return (new_x, new_y)
        return None

    def find_exit_from_shelf(self, warehouse: np.ndarray, all_robots: List['Robot']) -> Optional[Tuple[int, int]]:
        """Find the best exit path from a shelf position"""
        best_exit = None
        min_distance = float('inf')
        
        # Check all adjacent positions for a valid exit
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = self.x + dx, self.y + dy
            if (0 <= new_x < warehouse.shape[0] and 
                0 <= new_y < warehouse.shape[1] and 
                warehouse[new_x, new_y] == 0):  # Must be an aisle
                
                # Check if position is free of other robots
                position_free = True
                for robot in all_robots:
                    if robot is not self and robot.x == new_x and robot.y == new_y:
                        position_free = False
                        break
                
                if position_free:
                    # If we have a target, prefer the exit that's closer to it
                    if self.target_x is not None and self.target_y is not None:
                        distance = abs(new_x - self.target_x) + abs(new_y - self.target_y)
                        if distance < min_distance:
                            min_distance = distance
                            best_exit = (new_x, new_y)
                    else:
                        # If no target, take the first available exit
                        return (new_x, new_y)
        
        return best_exit

    def is_stuck(self) -> bool:
        """Check if robot is stuck in the same position"""
        current_pos = (self.x, self.y)
        is_stuck = (current_pos == self.last_position and 
                   self.total_wait_time > self.deadlock_threshold)
        if is_stuck:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.backup_attempts = 0
        self.last_position = current_pos
        return is_stuck

    def try_resolve_deadlock(self, warehouse: np.ndarray, all_robots: List['Robot']) -> bool:
        """Try various strategies to resolve deadlock"""
        print(f"Robot at ({self.x}, {self.y}) attempting to resolve deadlock")
        
        # If we're stuck for too long, try more aggressive strategies
        if self.stuck_count >= 3:
            # Try backing up in different directions
            backup_pos = self.find_backup_position(warehouse, all_robots)
            if backup_pos:
                print(f"Robot at ({self.x}, {self.y}) backing up to {backup_pos}")
                self.x, self.y = backup_pos
                self.path = []  # Clear path to recalculate from new position
                if self.target_x is not None:
                    self.path = find_shortest_path(warehouse, (self.x, self.y),
                                                 (self.target_x, self.target_y))
                self.total_wait_time = 0
                self.stuck_count = 0
                return True
                
            # If still stuck, temporarily clear target and become idle
            if self.stuck_count >= 5:
                print(f"Robot at ({self.x}, {self.y}) clearing target due to severe deadlock")
                self.target_x = None
                self.target_y = None
                self.path = []
                self.status = RobotStatus.IDLE
                self.total_wait_time = 0
                self.stuck_count = 0
                return True
        
        return False

    def find_backup_position(self, warehouse: np.ndarray, all_robots: List['Robot']) -> Optional[Tuple[int, int]]:
        """Find a position to back up to when stuck"""
        # Try different directions based on previous attempts
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        start_idx = self.backup_attempts % 4
        self.backup_attempts += 1
        
        # Check directions in order starting from our next attempt
        for i in range(4):
            dx, dy = directions[(start_idx + i) % 4]
            new_x, new_y = self.x + dx, self.y + dy
            
            if (0 <= new_x < warehouse.shape[0] and 
                0 <= new_y < warehouse.shape[1] and 
                warehouse[new_x, new_y] == 0):  # Must be an aisle
                
                # Check if position is free of other robots
                position_free = True
                for robot in all_robots:
                    if robot is not self and robot.x == new_x and robot.y == new_y:
                        position_free = False
                        break
                
                if position_free:
                    return (new_x, new_y)
        return None

    def is_valid_shelf_entry(self, warehouse: np.ndarray, shelf_x: int, shelf_y: int, from_x: int, from_y: int) -> bool:
        """Check if attempting to enter shelf from a valid adjacent aisle"""
        # Must be moving from an aisle to a shelf
        if warehouse[from_x, from_y] != 0 or warehouse[shelf_x, shelf_y] != 1:
            return False
            
        # Must be directly adjacent (no diagonal movement)
        dx = abs(shelf_x - from_x)
        dy = abs(shelf_y - from_y)
        if dx + dy != 1:  # Must move exactly one step
            return False
            
        return True

    def update_position(self, warehouse: np.ndarray, all_robots: List['Robot']) -> bool:
        """Update robot position based on current path and turning requirements"""
        # Track shelf entry point and entry direction
        if not self.path:
            return False
            
        next_pos = self.path[0]
        # Validate shelf entry if attempting to move onto a shelf
        if (warehouse[next_pos[0], next_pos[1]] == 1 and 
            warehouse[self.x, self.y] == 0):
            # Record entry point and rotation
            self.shelf_entry_point = (self.x, self.y)
            # Calculate and store entry rotation
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            if dx > 0:
                self.shelf_entry_rotation = 0  # Entered facing right
            elif dx < 0:
                self.shelf_entry_rotation = 180  # Entered facing left
            elif dy > 0:
                self.shelf_entry_rotation = 90  # Entered facing down
            else:
                self.shelf_entry_rotation = 270  # Entered facing up
            print(f"Robot at ({self.x}, {self.y}) recording shelf entry point and rotation {self.shelf_entry_rotation}°")

        # Check if we're on a shelf
        if warehouse[self.x, self.y] == 1:
            if self.shelf_wait == 0:
                self.shelf_wait = 3  # Set initial wait time for item retrieval
                print(f"Robot at ({self.x}, {self.y}) starting shelf operation")
                return False
            elif self.shelf_wait > 0:
                self.shelf_wait -= 1
                print(f"Robot at ({self.x}, {self.y}) retrieving item: {self.shelf_wait} steps left")
                if self.shelf_wait == 0:
                    # After retrieval, immediately turn 180 degrees and set path to workstation
                    print(f"Robot at ({self.x}, {self.y}) item retrieved, turning 180 degrees")
                    # Immediate 180-degree turn
                    self.rotation = (self.shelf_entry_rotation + 180) % 360
                    # Set path: first to entry point, then to workstation
                    if self.shelf_entry_point and self.target_x is not None:
                        print(f"Robot at ({self.x}, {self.y}) planning path to workstation via entry point")
                        # First move to entry point
                        self.path = [self.shelf_entry_point]
                        # Then find path to workstation
                        next_path = find_shortest_path(warehouse, self.shelf_entry_point,
                                                     (self.target_x, self.target_y))
                        if next_path:
                            self.path.extend(next_path[1:])
                    return False
                return False

        # Normal movement logic
        if not self.path:
            print(f"Robot at ({self.x}, {self.y}) has no path")
            return False

        # Remove any points in path that match current position
        while self.path and self.path[0] == (self.x, self.y):
            self.path.pop(0)
            print(f"Skipping current position in path")
            
        if not self.path:
            return False

        next_x, next_y = self.path[0]
        print(f"Robot at ({self.x}, {self.y}) moving to ({next_x}, {next_y})")
        
        # Calculate required rotation based on next movement
        dx = next_x - self.x
        dy = next_y - self.y
        
        # Determine required rotation for movement direction
        required_rotation = None
        if dx > 0:
            required_rotation = 0  # Must face right
        elif dx < 0:
            required_rotation = 180  # Must face left
        elif dy > 0:
            required_rotation = 90  # Must face down
        elif dy < 0:
            required_rotation = 270  # Must face up

        # If not facing the correct direction, turn first
        if self.rotation != required_rotation:
            # Calculate shortest turning direction
            diff = (required_rotation - self.rotation) % 360
            if diff > 180:
                diff -= 360
            # Turn by 90 degrees each step
            if diff > 0:
                self.rotation = (self.rotation + 90) % 360
            else:
                self.rotation = (self.rotation - 90) % 360
            print(f"Robot at ({self.x}, {self.y}) turning to {self.rotation}°")
            return False
        
        # Move to next position
        old_x, old_y = self.x, self.y
        self.x, self.y = self.path.pop(0)
        print(f"Robot moved from ({old_x}, {old_y}) to ({self.x}, {self.y})")
        return True

    def find_immediate_exit(self, warehouse: np.ndarray, all_robots: List['Robot']) -> Optional[Tuple[int, int]]:
        """Find any immediately available exit from shelf, ignoring optimal paths"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)  # Randomize to avoid all robots trying same direction
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if (0 <= new_x < warehouse.shape[0] and 
                0 <= new_y < warehouse.shape[1] and 
                warehouse[new_x, new_y] == 0):
                
                # Check if position is free of other robots
                position_free = True
                for robot in all_robots:
                    if robot is not self and robot.x == new_x and robot.y == new_y:
                        position_free = False
                        break
                
                if position_free:
                    return (new_x, new_y)
        
        return None

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
    
    def is_valid_move(current: Tuple[int, int], next_pos: Tuple[int, int]) -> bool:
        """Check if move between positions is valid"""
        curr_x, curr_y = current
        next_x, next_y = next_pos
        
        # Check if moving to/from shelf is valid
        if warehouse[curr_x, curr_y] == 1 or warehouse[next_x, next_y] == 1:
            # Must be moving between shelf and adjacent aisle
            if warehouse[curr_x, curr_y] == 1 and warehouse[next_x, next_y] != 0:
                return False
            if warehouse[next_x, next_y] == 1 and warehouse[curr_x, curr_y] != 0:
                return False
            # Must be directly adjacent (no diagonal movement)
            dx = abs(next_x - curr_x)
            dy = abs(next_y - curr_y)
            if dx + dy != 1:
                return False
        
        return True
    
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
                # Verify the final step to shelf is valid
                if not is_valid_move(current, end):
                    continue
                came_from[end] = current
            break
            
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = current[0] + dx, current[1] + dy
            next_pos = (next_x, next_y)
            
            # Check if move is within bounds and valid
            if (0 <= next_x < rows and 0 <= next_y < cols and 
                (warehouse[next_x, next_y] == 0 or next_pos == end) and
                is_valid_move(current, next_pos)):
                
                new_cost = cost_so_far[current] + 1
                
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