import numpy as np
from enum import Enum
import heapq
from typing import List, Tuple, Optional, Set, Dict

class RobotStatus(Enum):
    IDLE = "idle"
    WORKING = "working"

class Robot:
    def __init__(self, x: int, y: int, collision_method: int = 1):
        self.x = x  # grid coordinates
        self.y = y
        self.status = RobotStatus.IDLE
        self.target_x = None
        self.target_y = None
        self.path = []  # List of coordinates to follow
        self.planned_path = []  # Path with time information for reservations
        self.rotation = 0  # Degrees (0: right, 90: up, 180: left, 270: down)
        self.turning_steps = 0  # Steps remaining in current turn
        self.waiting_time = 0  # Time to wait at current location
        self.is_at_workstation = False
        self.collision_wait_counter = 0  # Counter for waiting when collision detected
        self.collision_method = collision_method  # 1: waiting, 2: reservation

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

    def update_position(self, warehouse: np.ndarray, all_robots: List['Robot'] = None, reservation_table: Dict[Tuple[int, int, int], int] = None) -> bool:
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

        # Handle collisions based on method
        if all_robots:
            if self.collision_method == 1:  # Waiting method
                if self.is_robot_in_front(all_robots):
                    if self.collision_wait_counter < 10:
                        print(f"Robot at ({self.x}, {self.y}) waiting for collision: {self.collision_wait_counter}/10 time units")
                        self.collision_wait_counter += 1
                        self.waiting_time = 1
                        return False
                    else:
                        print(f"Robot at ({self.x}, {self.y}) waited 10 time units, proceeding through collision")
                        self.collision_wait_counter = 0  # Reset counter for next collision
            elif self.collision_method == 2:  # Reservation method
                # Check if next position is safe according to reservation table
                if not is_cell_safe(next_x, next_y, len(self.path), reservation_table, id(self)):
                    print(f"Robot at ({self.x}, {self.y}) detected reservation conflict, waiting")
                    self.waiting_time = 1
                    return False

        # Move to next position if no turning needed
        old_x, old_y = self.x, self.y
        self.x, self.y = self.path.pop(0)
        print(f"Robot moved from ({old_x}, {old_y}) to ({self.x}, {self.y})")
        return True

def is_directional_aisle(warehouse: np.ndarray, pos: Tuple[int, int]) -> Optional[Tuple[bool, int]]:
    """
    Determine if a position is in a directional aisle and which side it's on.
    
    Args:
        warehouse: The warehouse grid
        pos: Position to check (x, y)
        
    Returns:
        (is_right_side, direction) where:
        - is_right_side: True if position is on right side of aisle, False if on left side
        - direction: Direction of aisle travel (0: right, 1: down, 2: left, 3: up)
        Returns None if not in a 2-width aisle
    """
    x, y = pos
    rows, cols = warehouse.shape
    
    # Skip if position is not an aisle
    if warehouse[x, y] != 0:
        return None
    
    # Check for horizontal aisle (look north and south)
    north_wall = x == 0 or warehouse[x-1, y] == 1
    south_wall = x == rows-1 or warehouse[x+1, y] == 1
    
    if north_wall and south_wall:
        # Check if part of 2-width aisle by checking adjacent position
        if y > 0 and y < cols-1:
            # Check left and right
            left_is_aisle = warehouse[x, y-1] == 0
            right_is_aisle = warehouse[x, y+1] == 0
            
            # If aisle continues both left and right, check if there's another parallel aisle
            if left_is_aisle and right_is_aisle:
                # Check if there's a parallel aisle to the north or south
                parallel_aisle_north = x > 1 and warehouse[x-2, y] == 0
                parallel_aisle_south = x < rows-2 and warehouse[x+2, y] == 0
                
                if parallel_aisle_north:
                    # This is a horizontal 2-width aisle
                    # If we're in bottom row, we're on right side for eastbound
                    is_right_side = x > x-2  # We're in the southern aisle
                    return (is_right_side, 0)  # Eastbound
                    
                if parallel_aisle_south:
                    # This is a horizontal 2-width aisle
                    # If we're in top row, we're on right side for westbound
                    is_right_side = x < x+2  # We're in the northern aisle
                    return (is_right_side, 2)  # Westbound
    
    # Check for vertical aisle (look east and west)
    east_wall = y == cols-1 or warehouse[x, y+1] == 1
    west_wall = y == 0 or warehouse[x, y-1] == 1
    
    if east_wall and west_wall:
        # Check if part of 2-width aisle by checking adjacent position
        if x > 0 and x < rows-1:
            # Check up and down
            up_is_aisle = warehouse[x-1, y] == 0
            down_is_aisle = warehouse[x+1, y] == 0
            
            # If aisle continues both up and down, check if there's another parallel aisle
            if up_is_aisle and down_is_aisle:
                # Check if there's a parallel aisle to the east or west
                parallel_aisle_east = y < cols-2 and warehouse[x, y+2] == 0
                parallel_aisle_west = y > 1 and warehouse[x, y-2] == 0
                
                if parallel_aisle_east:
                    # This is a vertical 2-width aisle
                    # If we're in left column, we're on right side for southbound
                    is_right_side = y < y+2  # We're in the western aisle
                    return (is_right_side, 1)  # Southbound
                    
                if parallel_aisle_west:
                    # This is a vertical 2-width aisle
                    # If we're in right column, we're on right side for northbound
                    is_right_side = y > y-2  # We're in the eastern aisle
                    return (is_right_side, 3)  # Northbound
    
    # Not in a 2-width aisle or not able to determine direction
    return None

def find_shortest_path(warehouse: np.ndarray, start: Tuple[int, int], 
                      end: Tuple[int, int], workstations: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """A* pathfinding algorithm for robot routing with directional aisle constraints
    
    Args:
        warehouse: The warehouse grid
        start: The starting position (x, y)
        end: The target position (x, y)
        workstations: List of workstation positions to avoid unless they are the destination
    """
    if start == end:
        return []
        
    # If workstations not provided, use empty list
    if workstations is None:
        workstations = []
        
    # Create a set of workstations for faster lookup
    workstation_set = set(workstations)

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
            
            # Check directional aisle constraints
            # Skip if we're trying to move on the wrong side of a directional aisle
            current_aisle_info = is_directional_aisle(warehouse, current)
            next_aisle_info = is_directional_aisle(warehouse, next_pos)
            
            # If moving within the same directional aisle
            if current_aisle_info and next_aisle_info:
                current_is_right, current_direction = current_aisle_info
                next_is_right, next_direction = next_aisle_info
                
                # If on the same aisle with same direction
                if current_direction == next_direction:
                    # If we're on the left side (wrong side)
                    if not current_is_right:
                        # Only allow moving in opposite direction of the aisle
                        allowed_direction = (current_direction + 2) % 4
                        movement_direction = -1
                        
                        # Determine our movement direction
                        if dx == 1:  # Moving down
                            movement_direction = 1
                        elif dx == -1:  # Moving up
                            movement_direction = 3
                        elif dy == 1:  # Moving right
                            movement_direction = 0
                        elif dy == -1:  # Moving left
                            movement_direction = 2
                            
                        # If we're trying to move with the aisle flow on the wrong side, skip
                        if movement_direction != allowed_direction:
                            continue
            
            # Allow movement through valid positions
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

def build_reservation_table(robots: List[Robot]) -> Dict[Tuple[int, int, int], int]:
    """Build a reservation table for time-expanded A* pathfinding
    
    Args:
        robots: List of robots to build reservations for
        
    Returns:
        Dictionary mapping (x, y, t) to robot_id
    """
    reservation_table = {}
    for i, robot in enumerate(robots):
        if robot.planned_path:
            for t, pos in enumerate(robot.planned_path):
                reservation_table[(pos[0], pos[1], t)] = i
    return reservation_table

def update_reservation_table(reservation_table: Dict[Tuple[int, int, int], int], 
                           robot: Robot, robot_id: Optional[int] = None) -> None:
    """Update reservation table with a robot's planned path
    
    Args:
        reservation_table: The reservation table to update
        robot: The robot whose path to add
        robot_id: Optional robot ID (defaults to index in robots list)
    """
    if reservation_table is None or not robot.planned_path:
        return
        
    # Clear old reservations for this robot
    if robot_id is not None:
        to_remove = []
        for key, rid in reservation_table.items():
            if rid == robot_id:
                to_remove.append(key)
        for key in to_remove:
            del reservation_table[key]
    
    # Add new reservations
    for t, pos in enumerate(robot.planned_path):
        reservation_table[(pos[0], pos[1], t)] = robot_id if robot_id is not None else id(robot)

def is_cell_safe(x: int, y: int, t: int, reservation_table: Dict[Tuple[int, int, int], int],
                robot_id: Optional[int] = None) -> bool:
    """Check if a cell is safe to move into at a given time
    
    Args:
        x, y: Cell coordinates
        t: Time step
        reservation_table: Current reservation table
        robot_id: ID of robot making the check (to ignore its own reservations)
        
    Returns:
        True if cell is safe, False if reserved by another robot
    """
    if reservation_table is None:
        return True
        
    key = (x, y, t)
    if key not in reservation_table:
        return True
    return robot_id is not None and reservation_table[key] == robot_id

def time_expanded_a_star(warehouse: np.ndarray, start: Tuple[int, int], end: Tuple[int, int],
                        reservation_table: Dict[Tuple[int, int, int], int],
                        start_time: int = 0, robot_id: Optional[int] = None,
                        workstations: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """A* pathfinding in time-expanded graph with reservations
    
    Args:
        warehouse: The warehouse grid
        start: Starting position (x, y)
        end: Target position (x, y)
        reservation_table: Current reservation table
        start_time: Time step to start planning from
        robot_id: ID of robot making the plan
        workstations: List of workstation positions to avoid unless destination
        
    Returns:
        List of positions forming the path, or None if no path found
    """
    if start == end:
        return []
        
    if workstations is None:
        workstations = []
    workstation_set = set(workstations)
    
    rows, cols = warehouse.shape
    queue = [(0, start_time, start)]  # (priority, time, pos)
    came_from = {(start_time, start): None}
    cost_so_far = {(start_time, start): 0}
    
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    while queue:
        _, current_time, current = heapq.heappop(queue)
        
        if current == end:
            # Reconstruct path
            path = []
            current_key = (current_time, current)
            while current_key in came_from:
                path.append(current_key[1])
                current_key = came_from[current_key]
            path.reverse()
            return path
            
        # Try each neighbor
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x = current[0] + dx
            next_y = current[1] + dy
            next_time = current_time + 1
            
            # Check bounds and walls
            if not (0 <= next_x < rows and 0 <= next_y < cols):
                continue
            if warehouse[next_x, next_y] == 1:  # Wall/shelf
                continue
                
            # Skip workstations unless they are the destination
            next_pos = (next_x, next_y)
            if next_pos in workstation_set and next_pos != end:
                continue
                
            # Check reservations
            if not is_cell_safe(next_x, next_y, next_time, reservation_table, robot_id):
                continue
                
            # Check directional constraints
            aisle_info = is_directional_aisle(warehouse, next_pos)
            if aisle_info:
                is_right_side, direction = aisle_info
                # Determine if movement follows aisle direction
                movement_dir = None
                if dx > 0:
                    movement_dir = 1  # Down
                elif dx < 0:
                    movement_dir = 3  # Up
                elif dy > 0:
                    movement_dir = 0  # Right
                elif dy < 0:
                    movement_dir = 2  # Left
                    
                # Skip if moving against aisle direction
                if movement_dir != direction:
                    continue
            
            next_key = (next_time, next_pos)
            new_cost = cost_so_far[(current_time, current)] + 1
            
            if next_key not in cost_so_far or new_cost < cost_so_far[next_key]:
                cost_so_far[next_key] = new_cost
                priority = new_cost + heuristic(next_pos, end)
                heapq.heappush(queue, (priority, next_time, next_pos))
                came_from[next_key] = (current_time, current)
    
    return None  # No path found 