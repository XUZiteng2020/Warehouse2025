import numpy as np
from typing import List, Tuple, Optional
from robot import (
    Robot, 
    find_shortest_path, 
    RobotStatus, 
    build_reservation_table,
    update_reservation_table,
    time_expanded_a_star,
    is_cell_safe
)
import random
from workstation import generate_workstation_positions

class WarehouseManager:
    def __init__(self, warehouse: np.ndarray, num_workstations: int = 2, collision_method: int = 1):
        self.warehouse = warehouse
        self.rows, self.cols = warehouse.shape
        self.robots: List[Robot] = []
        self.num_workstations = num_workstations
        self.workstations = self._initialize_workstations()
        self.orders = np.zeros_like(warehouse)
        self.time_step = 0
        self.completed_jobs = 0
        self.is_playing = False
        self.targeted_orders = set()
        self.collision_method = collision_method
        self.reservation_table = None  # Initialize as None by default

    def _initialize_workstations(self) -> List[Tuple[int, int]]:
        """Initialize workstations using random placement on the left edge"""
        return generate_workstation_positions(
            num_workstations=self.num_workstations,
            warehouse_height=self.rows
        )

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
            self.robots.append(Robot(pos[0], pos[1], self.collision_method))
        
        # Initialize reservation table only if using method 2
        if self.collision_method == 2:
            self.reservation_table = build_reservation_table(self.robots)
        else:
            self.reservation_table = None

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
                    # Find path to order - use different methods based on collision method
                    if self.collision_method == 2:
                        path = time_expanded_a_star(
                            self.warehouse,
                            (robot.x, robot.y),
                            order_location,
                            self.reservation_table,
                            len(robot.planned_path)
                        )
                        if path:
                            robot.path = path
                            robot.planned_path = path
                            update_reservation_table(self.reservation_table, robot)
                    else:
                        path = find_shortest_path(
                            self.warehouse,
                            (robot.x, robot.y),
                            order_location,
                            self.workstations
                        )
                        if path:
                            robot.path = path
                    
                    if path:
                        robot.target_x, robot.target_y = order_location

            # Update robot position
            moved = robot.update_position(self.warehouse, self.robots, self.reservation_table)
            any_movement = any_movement or moved

            # Check if robot reached its target
            if robot.target_x is not None and robot.x == robot.target_x and robot.y == robot.target_y:
                if robot.status == RobotStatus.IDLE:
                    # Reached order location
                    self.orders[robot.x, robot.y] = 0
                    self.targeted_orders.discard((robot.x, robot.y))
                    robot.status = RobotStatus.WORKING
                    robot.waiting_time = 3
                    
                    # Assign path to workstation - use different methods based on collision method
                    workstation = self.get_random_workstation()
                    if self.collision_method == 2:
                        path = time_expanded_a_star(
                            self.warehouse,
                            (robot.x, robot.y),
                            workstation,
                            self.reservation_table,
                            len(robot.planned_path)
                        )
                        if path:
                            robot.path = path
                            robot.planned_path = path
                            update_reservation_table(self.reservation_table, robot)
                    else:
                        path = find_shortest_path(
                            self.warehouse,
                            (robot.x, robot.y),
                            workstation,
                            self.workstations
                        )
                        if path:
                            robot.path = path
                    
                    if path:
                        robot.target_x, robot.target_y = workstation
                
                elif robot.status == RobotStatus.WORKING and (robot.x, robot.y) in self.workstations:
                    # Reached workstation
                    robot.waiting_time = 3
                    robot.status = RobotStatus.IDLE
                    robot.target_x = robot.target_y = None
                    robot.path = []
                    if self.collision_method == 2:
                        robot.planned_path = []
                        update_reservation_table(self.reservation_table, robot)

                    self.completed_jobs += 1

        return any_movement

    def toggle_play(self):
        """Toggle between play and pause states"""
        self.is_playing = not self.is_playing 