import numpy as np
from enum import Enum
import heapq
import random
from typing import List, Tuple, Optional, Set, Dict

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
        self.waiting_for_robot = None  # Reference to robot this one is waiting for
        self.alternate_path_attempts = 0  # Count how many times we've tried to find alternate paths

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
                self.waiting_for_robot = other_robot  # Track which robot we're waiting for
                return True
        
        # If we're not waiting for any robot, clear the reference        
        self.waiting_for_robot = None
        return False

    def check_deadlock(self, all_robots: List['Robot']) -> List['Robot']:
        """Check if this robot is in a deadlock with another robot.
        Returns a list of robots involved in the deadlock, or an empty list if no deadlock."""
        if not self.waiting_for_robot:
            return []
            
        # Initialize the chain with this robot
        deadlock_chain = [self]
        current = self.waiting_for_robot
        
        # Follow the chain of waiting robots
        while current and current not in deadlock_chain:
            deadlock_chain.append(current)
            current = current.waiting_for_robot
            
        # If current is None, there's no cycle
        if not current:
            return []
            
        # If current is in the chain, we have a cycle
        # Return only the robots in the cycle
        cycle_start_index = deadlock_chain.index(current)
        return deadlock_chain[cycle_start_index:]
        
    def resolve_deadlock(self, deadlock_robots: List['Robot'], warehouse: np.ndarray, 
                        workstations: List[Tuple[int, int]]) -> None:
        """Resolve a deadlock by planning an alternative path for one of the robots"""
        if not deadlock_robots or not self in deadlock_robots:
            return
        
        # Filter robots that have targets
        robots_with_targets = [r for r in deadlock_robots if r.target_x is not None and r.target_y is not None]
        
        # If none of the robots have targets, just make them all wait different amounts of time
        if not robots_with_targets:
            print(f"No robots in deadlock have targets. Applying time-based resolution.")
            for i, robot in enumerate(deadlock_robots):
                # Assign different waiting times to break the cycle
                robot.waiting_time = 2 + i
                robot.waiting_for_robot = None
            return
        
        # Prefer to reroute this robot if it has a target, otherwise choose randomly from those with targets
        if self in robots_with_targets and (random.random() < 0.5 or self.alternate_path_attempts <= 3):
            robot_to_reroute = self
            self.alternate_path_attempts += 1
        else:
            robot_to_reroute = random.choice(robots_with_targets)
        
        print(f"Resolving deadlock involving {len(deadlock_robots)} robots. Rerouting robot at ({robot_to_reroute.x}, {robot_to_reroute.y})")
        
        # Gather positions of all robots to avoid
        positions_to_avoid = []
        for robot in deadlock_robots:
            if robot != robot_to_reroute:
                positions_to_avoid.append((robot.x, robot.y))
        
        # Find alternative path
        new_path = find_alternative_path(
            warehouse, 
            (robot_to_reroute.x, robot_to_reroute.y),
            (robot_to_reroute.target_x, robot_to_reroute.target_y),
            workstations,
            positions_to_avoid
        )
        
        if new_path:
            print(f"Found alternative path for robot at ({robot_to_reroute.x}, {robot_to_reroute.y}): {new_path}")
            robot_to_reroute.path = new_path
            robot_to_reroute.waiting_time = 0  # Reset waiting time
            robot_to_reroute.waiting_for_robot = None
        else:
            print(f"Could not find alternative path for robot at ({robot_to_reroute.x}, {robot_to_reroute.y})")
            # If we couldn't find an alternative path, just back off for a bit
            for i, robot in enumerate(deadlock_robots):
                # Assign different waiting times to break the cycle
                robot.waiting_time = 2 + i
                robot.waiting_for_robot = None

    def update_position(self, warehouse: np.ndarray, all_robots: List['Robot'] = None, 
                       workstations: List[Tuple[int, int]] = None) -> bool:
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
            # Check for deadlocks
            deadlock_robots = self.check_deadlock(all_robots)
            if deadlock_robots:
                print(f"Deadlock detected! Robot at ({self.x}, {self.y}) is in a deadlock with {len(deadlock_robots)-1} other robots")
                self.resolve_deadlock(deadlock_robots, warehouse, workstations)
                return False
            
            print(f"Robot at ({self.x}, {self.y}) waiting for 1 time unit due to robot in front")
            self.waiting_time = 1
            return False

        # Move to next position if no turning needed and no robot in front
        old_x, old_y = self.x, self.y
        self.x, self.y = self.path.pop(0)
        print(f"Robot moved from ({old_x}, {old_y}) to ({self.x}, {self.y})")
        # Reset alternate path attempts counter on successful move
        self.alternate_path_attempts = 0
        return True

def find_alternative_path(warehouse: np.ndarray, start: Tuple[int, int], 
                        end: Tuple[int, int], workstations: List[Tuple[int, int]] = None,
                        positions_to_avoid: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """A* pathfinding algorithm that avoids specific positions"""
    if start == end:
        return []
        
    # If workstations not provided, use empty list
    if workstations is None:
        workstations = []
    
    if positions_to_avoid is None:
        positions_to_avoid = []
        
    # Create sets for faster lookup
    workstation_set = set(workstations)
    avoid_set = set(positions_to_avoid)

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
            next_pos = (next_x, next_y)
            
            # Skip if out of bounds
            if not (0 <= next_x < rows and 0 <= next_y < cols):
                continue
                
            # Skip if it's a shelf (unless it's the destination)
            if warehouse[next_x, next_y] == 1 and next_pos != end:
                continue
                
            # Skip if it's a workstation that's not our destination
            if next_pos in workstation_set and next_pos != end:
                continue
                
            # Skip if it's a position we're trying to avoid
            if next_pos in avoid_set:
                continue
            
            # Allow movement through aisles and to the final shelf position
            new_cost = cost_so_far[current] + 1
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(end, next_pos)
                heapq.heappush(queue, (priority, next_pos))
                came_from[next_pos] = current
    
    if end not in came_from:
        print(f"No alternative path found from {start} to {end} avoiding {positions_to_avoid}")
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
    
    print(f"Found alternative path from {start} to {end}: {path}")
    return path

def find_shortest_path(warehouse: np.ndarray, start: Tuple[int, int], 
                      end: Tuple[int, int], workstations: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """A* pathfinding algorithm for robot routing
    
    Args:
        warehouse: The warehouse grid
        start: The starting position (x, y)
        end: The target position (x, y)
        workstations: List of workstation positions to avoid unless they are the destination
    """
    return find_alternative_path(warehouse, start, end, workstations) 