import numpy as np
from typing import List, Tuple, Optional
from robot import Robot, find_shortest_path, RobotStatus
import random

class WarehouseManager:
    def __init__(self, warehouse: np.ndarray):
        self.warehouse = warehouse
        self.rows, self.cols = warehouse.shape
        self.robots: List[Robot] = []
        self.workstations = self._initialize_workstations()
        self.orders = np.zeros_like(warehouse)
        self.time_step = 0
        self.completed_jobs = 0
        self.is_playing = False
        self.targeted_orders = set()  # Keep track of orders being targeted by robots

    def _initialize_workstations(self) -> List[Tuple[int, int]]:
        """Initialize workstations at the leftmost end of each aisle"""
        workstations = []
        for i in range(self.rows):
            if self.warehouse[i, 0] == 0:  # If it's an aisle
                workstations.append((i, 0))
        return workstations

    def initialize_robots(self, n: int):
        """Initialize n robots at random valid positions"""
        self.robots.clear()
        self.targeted_orders.clear()
        available_positions = [(i, j) for i in range(self.rows) for j in range(self.cols)
                             if self.warehouse[i, j] == 0 and (i, j) not in self.workstations]
        
        if n > len(available_positions):
            n = len(available_positions)
        
        positions = random.sample(available_positions, n)
        for pos in positions:
            self.robots.append(Robot(pos[0], pos[1]))

    def update_orders(self, orders: np.ndarray):
        """Update the order matrix"""
        self.orders = orders.copy()
        self.targeted_orders.clear()  # Reset targeted orders when new orders are generated

    def get_random_order_location(self) -> Optional[Tuple[int, int]]:
        """Get a random untargeted order location"""
        order_locations = [(i, j) for i in range(self.rows) for j in range(self.cols)
                          if self.orders[i, j] == 1 and (i, j) not in self.targeted_orders]
        if not order_locations:
            return None
        
        chosen_location = random.choice(order_locations)
        self.targeted_orders.add(chosen_location)
        print(f"Assigning order at {chosen_location} to robot")
        return chosen_location

    def get_random_workstation(self) -> Tuple[int, int]:
        """Get a random workstation location"""
        return random.choice(self.workstations)

    def update(self) -> bool:
        """Update the state of all robots and return whether any robot moved"""
        if not self.is_playing:
            return False

        any_movement = False
        self.time_step += 1

        for robot in self.robots:
            # Check if robot needs a new task
            if robot.status == RobotStatus.IDLE and not robot.path:
                order_location = self.get_random_order_location()
                if order_location:
                    # Find path to order
                    path = find_shortest_path(self.warehouse, (robot.x, robot.y), order_location)
                    if path:
                        robot.path = path
                        robot.target_x, robot.target_y = order_location
                        print(f"Robot at ({robot.x}, {robot.y}) assigned to order at {order_location}")

            # Update robot position
            moved = robot.update_position(self.warehouse)
            any_movement = any_movement or moved

            # Check if robot reached its target
            if robot.target_x is not None and robot.x == robot.target_x and robot.y == robot.target_y:
                if robot.status == RobotStatus.IDLE:
                    # Reached order location
                    print(f"Robot reached order at ({robot.x}, {robot.y})")
                    self.orders[robot.x, robot.y] = 0
                    # Remove from targeted orders when picked up
                    self.targeted_orders.discard((robot.x, robot.y))
                    robot.status = RobotStatus.WORKING
                    robot.waiting_time = 3  # Wait 3 seconds
                    
                    # Assign path to workstation
                    workstation = self.get_random_workstation()
                    path = find_shortest_path(self.warehouse, (robot.x, robot.y), workstation)
                    if path:
                        robot.path = path
                        robot.target_x, robot.target_y = workstation
                        print(f"Robot assigned to workstation at {workstation}")
                
                elif robot.status == RobotStatus.WORKING and (robot.x, robot.y) in self.workstations:
                    # Reached workstation
                    print(f"Robot completed job at workstation ({robot.x}, {robot.y})")
                    robot.waiting_time = 3  # Wait 3 seconds
                    robot.status = RobotStatus.IDLE
                    robot.target_x = robot.target_y = None
                    robot.path = []
                    self.completed_jobs += 1

        return any_movement

    def toggle_play(self):
        """Toggle between play and pause states"""
        self.is_playing = not self.is_playing
        print(f"Simulation {'started' if self.is_playing else 'paused'}") 