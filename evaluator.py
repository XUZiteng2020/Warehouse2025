import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import tqdm
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import heapq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import multiprocessing as mp
from functools import partial

# Import warehouse simulation components
from warehouse_manager import WarehouseManager
from robot import RobotStatus, find_shortest_path
from order_generation import uniform_order_distribution

# Original pathfinding function without directional constraint (for comparison)
def find_path_without_directional_constraint(warehouse: np.ndarray, start: Tuple[int, int], 
                      end: Tuple[int, int], workstations: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """A* pathfinding algorithm without directional constraints (for comparison)"""
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
    
    print(f"Found path from {start} to {end} (without directional constraint)")
    return path

def run_simulation_until_complete(warehouse: np.ndarray, 
                                robot_count: int, 
                                order_probability: float = 0.3,
                                max_steps: int = 5000) -> Tuple[int, int]:
    """
    Run a simulation until all orders are completed or max steps is reached.
    
    Args:
        warehouse: The warehouse grid
        robot_count: Number of robots to use
        order_probability: Probability of an order at each shelf
        max_steps: Maximum simulation steps before timeout
    
    Returns:
        Tuple of (time_steps, completed_jobs)
    """
    # Create warehouse manager
    warehouse_manager = WarehouseManager(warehouse)
    
    # Initialize robots
    warehouse_manager.initialize_robots(robot_count)
    
    # Generate orders
    orders = uniform_order_distribution(order_probability, warehouse)
    warehouse_manager.update_orders(orders)
    
    # Count total orders
    total_orders = np.sum(orders)
    
    # Start simulation
    warehouse_manager.is_playing = True
    
    # Run until all orders complete or max steps reached
    steps = 0
    while steps < max_steps:
        steps += 1
        
        # Update warehouse state
        movement = warehouse_manager.update()
        
        # Check if all orders are completed
        if warehouse_manager.completed_jobs >= total_orders:
            return steps, warehouse_manager.completed_jobs
            
        # If no movement and no pending orders, we're done
        if not movement and np.sum(warehouse_manager.orders) == 0:
            return steps, warehouse_manager.completed_jobs
    
    # If we reached max steps, return current state
    return max_steps, warehouse_manager.completed_jobs

def analyze_completion_time_distribution(
    num_samples: int = 10,
    order_probability: float = 0.3,
    max_steps: int = 10000
) -> None:
    """
    Analyze the distribution of completion times across multiple order distributions.
    Uses the median number of robots in the valid range.
    
    Args:
        num_samples: Number of different order distributions to sample
        order_probability: Probability of an order at each shelf
        max_steps: Maximum simulation steps before timeout
    """
    # Load a random warehouse layout
    warehouse_dir = 'warehouse_data_files'
    warehouse_files = [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')]
    random_warehouse_file = os.path.join(warehouse_dir, random.choice(warehouse_files))
    warehouse = np.loadtxt(random_warehouse_file)
    
    # Calculate the valid robot range
    num_shelves = np.sum(warehouse == 1)
    available_aisles = np.sum(warehouse == 0)
    
    # Calculate min and max robots
    min_robots = 1
    max_robots = min(int(np.sqrt(num_shelves)), available_aisles - 10)  # Leave some space
    
    # Use the median number of robots
    median_robots = (min_robots + max_robots) // 2
    print(f"Running simulations with {median_robots} robots")
    print(f"Valid robot range: {min_robots} to {max_robots}")
    print(f"Warehouse has {num_shelves} shelves and {available_aisles} aisle cells")
    
    # Collect completion times
    completion_times = []
    completion_jobs = []
    
    print(f"Starting {num_samples} simulations...")
    for i in tqdm.tqdm(range(num_samples)):
        # Generate a new order distribution for each sample
        steps, jobs = run_simulation_until_complete(
            warehouse, 
            median_robots, 
            order_probability,
            max_steps
        )
        completion_times.append(steps)
        completion_jobs.append(jobs)
    
    # Calculate statistics
    avg_time = np.mean(completion_times)
    median_time = np.median(completion_times)
    min_time = np.min(completion_times)
    max_time = np.max(completion_times)
    
    avg_jobs = np.mean(completion_jobs)
    min_jobs = np.min(completion_jobs)
    max_jobs = np.max(completion_jobs)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(completion_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_time:.1f}')
    plt.axvline(median_time, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_time:.1f}')
    
    plt.title(f'Distribution of Completion Times\n{median_robots} robots, {order_probability:.1f} order probability')
    plt.xlabel('Time Steps to Complete All Orders')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig('completion_time_distribution.png', dpi=300, bbox_inches='tight')
    
    # Print statistics
    print("\nCompletion Time Statistics:")
    print(f"Average time: {avg_time:.2f} steps")
    print(f"Median time: {median_time:.2f} steps")
    print(f"Min time: {min_time} steps")
    print(f"Max time: {max_time} steps")
    print(f"Timeout cases: {completion_times.count(max_steps)} / {num_samples}")
    
    print("\nCompletion Jobs Statistics:")
    print(f"Average jobs: {avg_jobs:.2f}")
    print(f"Min jobs: {min_jobs}")
    print(f"Max jobs: {max_jobs}")
    
    plt.show()

def analyze_robot_count_impact(
    num_samples: int = 50,
    order_probability: float = 0.3,
    max_steps: int = 10000
) -> None:
    """
    Analyze the impact of different robot counts on completion time.
    
    Args:
        num_samples: Number of order distributions to sample for each robot count
        order_probability: Probability of an order at each shelf
        max_steps: Maximum simulation steps before timeout
    """
    # Load a random warehouse layout
    warehouse_dir = 'warehouse_data_files'
    warehouse_files = [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')]
    random_warehouse_file = os.path.join(warehouse_dir, random.choice(warehouse_files))
    warehouse = np.loadtxt(random_warehouse_file)
    
    # Calculate the valid robot range
    num_shelves = np.sum(warehouse == 1)
    available_aisles = np.sum(warehouse == 0)
    
    # Calculate min and max robots
    min_robots = 1
    max_robot_limit = min(int(np.sqrt(num_shelves)), available_aisles - 10)
    
    # Generate robot counts to test (using 7 points spread across the range)
    robot_counts = [
        max(1, int(max_robot_limit * 0.1)),  # 10% of max
        max(1, int(max_robot_limit * 0.25)),  # 25% of max
        max(1, int(max_robot_limit * 0.5)),   # 50% of max
        max(1, int(max_robot_limit * 0.75)),  # 75% of max
        max(1, int(max_robot_limit * 0.9)),   # 90% of max
        max_robot_limit                       # 100% of max
    ]
    
    # Ensure unique values and sort
    robot_counts = sorted(set(robot_counts))
    
    print(f"Testing robot counts: {robot_counts}")
    print(f"Warehouse has {num_shelves} shelves and {available_aisles} aisle cells")
    print(f"Maximum robot count by sqrt rule: {int(np.sqrt(num_shelves))}")
    
    # Store results for each robot count
    results = {
        'avg_time': [],
        'median_time': [],
        'min_time': [],
        'max_time': [],
        'timeout_rate': [],
        'avg_jobs': [],
        'std_time': []
    }
    
    # Generate a fixed set of order distributions to use for all robot counts
    print("Generating order distributions...")
    order_distributions = []
    for _ in range(num_samples):
        orders = uniform_order_distribution(order_probability, warehouse)
        order_distributions.append(orders)
    
    # Run simulations for each robot count
    for robot_count in robot_counts:
        print(f"\nTesting with {robot_count} robots...")
        
        completion_times = []
        completion_jobs = []
        
        for i, orders in tqdm.tqdm(enumerate(order_distributions), total=len(order_distributions)):
            # Create warehouse manager
            warehouse_manager = WarehouseManager(warehouse)
            warehouse_manager.initialize_robots(robot_count)
            warehouse_manager.update_orders(orders.copy())
            
            # Count total orders
            total_orders = np.sum(orders)
            
            # Start simulation
            warehouse_manager.is_playing = True
            
            # Run until all orders complete or max steps reached
            steps = 0
            while steps < max_steps:
                steps += 1
                
                # Update warehouse state
                movement = warehouse_manager.update()
                
                # Check if all orders are completed
                if warehouse_manager.completed_jobs >= total_orders:
                    break
                    
                # If no movement and no pending orders, we're done
                if not movement and np.sum(warehouse_manager.orders) == 0:
                    break
            
            completion_times.append(steps)
            completion_jobs.append(warehouse_manager.completed_jobs)
        
        # Calculate statistics
        avg_time = np.mean(completion_times)
        median_time = np.median(completion_times)
        min_time = np.min(completion_times)
        max_time = np.max(completion_times)
        std_time = np.std(completion_times)
        
        avg_jobs = np.mean(completion_jobs)
        timeout_rate = completion_times.count(max_steps) / len(completion_times)
        
        # Store results
        results['avg_time'].append(avg_time)
        results['median_time'].append(median_time)
        results['min_time'].append(min_time)
        results['max_time'].append(max_time)
        results['timeout_rate'].append(timeout_rate)
        results['avg_jobs'].append(avg_jobs)
        results['std_time'].append(std_time)
        
        print(f"Average time: {avg_time:.2f} steps")
        print(f"Median time: {median_time:.2f} steps")
        print(f"Min time: {min_time} steps")
        print(f"Max time: {max_time} steps")
        print(f"Std deviation: {std_time:.2f} steps")
        print(f"Timeout rate: {timeout_rate:.2%}")
        print(f"Average jobs: {avg_jobs:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot average time with error bars
    plt.errorbar(robot_counts, results['avg_time'], yerr=results['std_time'], 
                fmt='o-', capsize=5, label='Avg Time ± Std Dev')
    
    # Plot median time
    plt.plot(robot_counts, results['median_time'], 's--', label='Median Time')
    
    # Add min/max band
    plt.fill_between(robot_counts, results['min_time'], results['max_time'], 
                    alpha=0.2, label='Min-Max Range')
    
    plt.title('Impact of Robot Count on Completion Time')
    plt.xlabel('Number of Robots')
    plt.ylabel('Time Steps to Complete All Orders')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add second y-axis for timeout rate
    ax2 = plt.gca().twinx()
    ax2.plot(robot_counts, [r * 100 for r in results['timeout_rate']], 'r--', label='Timeout Rate')
    ax2.set_ylabel('Timeout Rate (%)')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add second legend
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('robot_count_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot secondary metrics
    plt.figure(figsize=(10, 6))
    plt.plot(robot_counts, results['avg_jobs'], 'o-', label='Avg Completed Jobs')
    
    plt.title('Completed Jobs by Robot Count')
    plt.xlabel('Number of Robots')
    plt.ylabel('Average Completed Jobs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('completed_jobs_by_robot_count.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_directional_aisles(
    num_samples: int = 50,
    order_probability: float = 0.3,
    max_steps: int = 5000
) -> None:
    """
    Compare the performance of directional aisles vs. non-directional aisles.
    
    Args:
        num_samples: Number of order distributions to sample
        order_probability: Probability of an order at each shelf
        max_steps: Maximum simulation steps before timeout
    """
    from robot import find_shortest_path as directional_path_finder
    
    # Load a random warehouse layout
    warehouse_dir = 'warehouse_data_files'
    warehouse_files = [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')]
    random_warehouse_file = os.path.join(warehouse_dir, random.choice(warehouse_files))
    warehouse = np.loadtxt(random_warehouse_file)
    
    # Calculate the valid robot range
    num_shelves = np.sum(warehouse == 1)
    available_aisles = np.sum(warehouse == 0)
    
    # Calculate min and max robots
    min_robots = 1
    max_robot_limit = min(int(np.sqrt(num_shelves)), available_aisles - 10)
    
    # Define robot count levels to test (low, medium, high)
    robot_counts = [
        max(1, int(max_robot_limit * 0.25)),      # Low (25%)
        max(1, int(max_robot_limit * 0.5)),       # Medium (50%)
        max(1, int(max_robot_limit * 0.75))       # High (75%)
    ]
    
    print(f"Testing robot counts: {robot_counts}")
    print(f"Warehouse has {num_shelves} shelves and {available_aisles} aisle cells")
    
    # Generate a fixed set of order distributions to use for all tests
    print("Generating order distributions...")
    order_distributions = []
    for _ in range(num_samples):
        orders = uniform_order_distribution(order_probability, warehouse)
        order_distributions.append(orders)
    
    # Structure to store results
    results = {
        'directional': {count: {'times': [], 'jobs': [], 'conflicts': []} for count in robot_counts},
        'non_directional': {count: {'times': [], 'jobs': [], 'conflicts': []} for count in robot_counts}
    }
    
    # First run simulations with directional aisles
    print("\n\n--- Testing with Directional Aisles ---")
    
    for robot_count in robot_counts:
        print(f"\nTesting {robot_count} robots with directional aisles...")
        
        for i, orders in tqdm.tqdm(enumerate(order_distributions), total=len(order_distributions)):
            # Keep a local copy of the original find_shortest_path function
            original_path_finder = find_shortest_path
            
            # Set the robot module to use directional pathfinding
            import robot
            robot.find_shortest_path = directional_path_finder
            
            # Create warehouse manager
            warehouse_manager = WarehouseManager(warehouse)
            warehouse_manager.initialize_robots(robot_count)
            warehouse_manager.update_orders(orders.copy())
            
            # Count total orders
            total_orders = np.sum(orders)
            
            # Start simulation
            warehouse_manager.is_playing = True
            
            # Track conflict count
            conflict_count = 0
            
            # Run until all orders complete or max steps reached
            steps = 0
            while steps < max_steps:
                steps += 1
                
                # Update warehouse state
                movement = warehouse_manager.update()
                
                # Count robot conflicts (robots waiting for other robots)
                for robot in warehouse_manager.robots:
                    if hasattr(robot, 'waiting_for_robot') and robot.waiting_for_robot is not None:
                        conflict_count += 1
                
                # Check if all orders are completed
                if warehouse_manager.completed_jobs >= total_orders:
                    break
                    
                # If no movement and no pending orders, we're done
                if not movement and np.sum(warehouse_manager.orders) == 0:
                    break
            
            results['directional'][robot_count]['times'].append(steps)
            results['directional'][robot_count]['jobs'].append(warehouse_manager.completed_jobs)
            results['directional'][robot_count]['conflicts'].append(conflict_count)
    
    # Now run simulations with non-directional aisles
    print("\n\n--- Testing with Non-Directional Aisles ---")
    
    for robot_count in robot_counts:
        print(f"\nTesting {robot_count} robots with non-directional aisles...")
        
        for i, orders in tqdm.tqdm(enumerate(order_distributions), total=len(order_distributions)):
            # Replace the pathfinding function with the non-directional version
            import robot
            robot.find_shortest_path = find_path_without_directional_constraint
            
            # Create warehouse manager
            warehouse_manager = WarehouseManager(warehouse)
            warehouse_manager.initialize_robots(robot_count)
            warehouse_manager.update_orders(orders.copy())
            
            # Count total orders
            total_orders = np.sum(orders)
            
            # Start simulation
            warehouse_manager.is_playing = True
            
            # Track conflict count
            conflict_count = 0
            
            # Run until all orders complete or max steps reached
            steps = 0
            while steps < max_steps:
                steps += 1
                
                # Update warehouse state
                movement = warehouse_manager.update()
                
                # Count robot conflicts (robots waiting for other robots)
                for robot in warehouse_manager.robots:
                    if hasattr(robot, 'waiting_for_robot') and robot.waiting_for_robot is not None:
                        conflict_count += 1
                
                # Check if all orders are completed
                if warehouse_manager.completed_jobs >= total_orders:
                    break
                    
                # If no movement and no pending orders, we're done
                if not movement and np.sum(warehouse_manager.orders) == 0:
                    break
            
            results['non_directional'][robot_count]['times'].append(steps)
            results['non_directional'][robot_count]['jobs'].append(warehouse_manager.completed_jobs)
            results['non_directional'][robot_count]['conflicts'].append(conflict_count)
    
    # Restore the original path finder
    import robot
    robot.find_shortest_path = directional_path_finder
    
    # Calculate summary statistics
    summary = {
        'directional': {count: {
            'avg_time': np.mean(results['directional'][count]['times']),
            'std_time': np.std(results['directional'][count]['times']),
            'avg_jobs': np.mean(results['directional'][count]['jobs']),
            'avg_conflicts': np.mean(results['directional'][count]['conflicts']),
            'timeout_rate': results['directional'][count]['times'].count(max_steps) / len(results['directional'][count]['times'])
        } for count in robot_counts},
        'non_directional': {count: {
            'avg_time': np.mean(results['non_directional'][count]['times']),
            'std_time': np.std(results['non_directional'][count]['times']),
            'avg_jobs': np.mean(results['non_directional'][count]['jobs']),
            'avg_conflicts': np.mean(results['non_directional'][count]['conflicts']),
            'timeout_rate': results['non_directional'][count]['times'].count(max_steps) / len(results['non_directional'][count]['times'])
        } for count in robot_counts}
    }
    
    # Print summary statistics
    print("\n\n--- Summary Statistics ---")
    print("\nDirectional Aisles:")
    for count in robot_counts:
        print(f"\n  {count} Robots:")
        print(f"    Avg Time: {summary['directional'][count]['avg_time']:.2f} ± {summary['directional'][count]['std_time']:.2f}")
        print(f"    Avg Jobs: {summary['directional'][count]['avg_jobs']:.2f}")
        print(f"    Avg Conflicts: {summary['directional'][count]['avg_conflicts']:.2f}")
        print(f"    Timeout Rate: {summary['directional'][count]['timeout_rate']:.2%}")
    
    print("\nNon-Directional Aisles:")
    for count in robot_counts:
        print(f"\n  {count} Robots:")
        print(f"    Avg Time: {summary['non_directional'][count]['avg_time']:.2f} ± {summary['non_directional'][count]['std_time']:.2f}")
        print(f"    Avg Jobs: {summary['non_directional'][count]['avg_jobs']:.2f}")
        print(f"    Avg Conflicts: {summary['non_directional'][count]['avg_conflicts']:.2f}")
        print(f"    Timeout Rate: {summary['non_directional'][count]['timeout_rate']:.2%}")
    
    # Plot comparative results
    # 1. Time comparison
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(robot_counts))
    width = 0.35
    
    dir_times = [summary['directional'][count]['avg_time'] for count in robot_counts]
    dir_std = [summary['directional'][count]['std_time'] for count in robot_counts]
    
    non_dir_times = [summary['non_directional'][count]['avg_time'] for count in robot_counts]
    non_dir_std = [summary['non_directional'][count]['std_time'] for count in robot_counts]
    
    bar1 = plt.bar(x - width/2, dir_times, width, yerr=dir_std, label='Directional Aisles', 
                  color='royalblue', alpha=0.7, capsize=5)
    bar2 = plt.bar(x + width/2, non_dir_times, width, yerr=non_dir_std, label='Non-Directional Aisles', 
                  color='orangered', alpha=0.7, capsize=5)
    
    plt.xlabel('Number of Robots')
    plt.ylabel('Average Completion Time (steps)')
    plt.title('Directional vs. Non-Directional Aisles: Completion Time')
    plt.xticks(x, robot_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add percentage improvement labels
    for i, (d_time, nd_time) in enumerate(zip(dir_times, non_dir_times)):
        if d_time < nd_time:
            pct_improvement = ((nd_time - d_time) / nd_time) * 100
            text = f"{pct_improvement:.1f}% faster"
            plt.annotate(text, xy=(x[i], min(d_time, nd_time) - 50), ha='center', va='top',
                         color='green', fontweight='bold')
        elif d_time > nd_time:
            pct_degradation = ((d_time - nd_time) / nd_time) * 100
            text = f"{pct_degradation:.1f}% slower"
            plt.annotate(text, xy=(x[i], min(d_time, nd_time) - 50), ha='center', va='top',
                         color='red', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('directional_vs_nondirectional_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Conflict comparison
    plt.figure(figsize=(14, 8))
    
    dir_conflicts = [summary['directional'][count]['avg_conflicts'] for count in robot_counts]
    non_dir_conflicts = [summary['non_directional'][count]['avg_conflicts'] for count in robot_counts]
    
    bar1 = plt.bar(x - width/2, dir_conflicts, width, label='Directional Aisles', 
                  color='royalblue', alpha=0.7)
    bar2 = plt.bar(x + width/2, non_dir_conflicts, width, label='Non-Directional Aisles', 
                  color='orangered', alpha=0.7)
    
    plt.xlabel('Number of Robots')
    plt.ylabel('Average Number of Conflicts')
    plt.title('Directional vs. Non-Directional Aisles: Robot Conflicts')
    plt.xticks(x, robot_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add percentage improvement labels
    for i, (d_conf, nd_conf) in enumerate(zip(dir_conflicts, non_dir_conflicts)):
        if d_conf < nd_conf:
            pct_improvement = ((nd_conf - d_conf) / (nd_conf if nd_conf > 0 else 1)) * 100
            text = f"{pct_improvement:.1f}% fewer"
            plt.annotate(text, xy=(x[i], min(d_conf, nd_conf) - 5), ha='center', va='top',
                         color='green', fontweight='bold')
        elif d_conf > nd_conf:
            pct_degradation = ((d_conf - nd_conf) / (nd_conf if nd_conf > 0 else 1)) * 100
            text = f"{pct_degradation:.1f}% more"
            plt.annotate(text, xy=(x[i], min(d_conf, nd_conf) - 5), ha='center', va='top',
                         color='red', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('directional_vs_nondirectional_conflicts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Jobs comparison
    plt.figure(figsize=(14, 8))
    
    dir_jobs = [summary['directional'][count]['avg_jobs'] for count in robot_counts]
    non_dir_jobs = [summary['non_directional'][count]['avg_jobs'] for count in robot_counts]
    
    bar1 = plt.bar(x - width/2, dir_jobs, width, label='Directional Aisles', 
                  color='royalblue', alpha=0.7)
    bar2 = plt.bar(x + width/2, non_dir_jobs, width, label='Non-Directional Aisles', 
                  color='orangered', alpha=0.7)
    
    plt.xlabel('Number of Robots')
    plt.ylabel('Average Completed Jobs')
    plt.title('Directional vs. Non-Directional Aisles: Jobs Completed')
    plt.xticks(x, robot_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add percentage improvement labels
    for i, (d_jobs, nd_jobs) in enumerate(zip(dir_jobs, non_dir_jobs)):
        if d_jobs > nd_jobs:
            pct_improvement = ((d_jobs - nd_jobs) / (nd_jobs if nd_jobs > 0 else 1)) * 100
            text = f"{pct_improvement:.1f}% more"
            plt.annotate(text, xy=(x[i], min(d_jobs, nd_jobs) - 0.5), ha='center', va='top',
                         color='green', fontweight='bold')
        elif d_jobs < nd_jobs:
            pct_degradation = ((nd_jobs - d_jobs) / (d_jobs if d_jobs > 0 else 1)) * 100
            text = f"{pct_degradation:.1f}% fewer"
            plt.annotate(text, xy=(x[i], min(d_jobs, nd_jobs) - 0.5), ha='center', va='top',
                         color='red', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('directional_vs_nondirectional_jobs.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_parallel_simulations(params: Tuple) -> Tuple[int, int, Dict]:
    """
    Run a single simulation with given parameters. Designed for parallel processing.
    
    Args:
        params: Tuple containing (warehouse, robot_count, order_probability, max_steps)
        
    Returns:
        Tuple of (time_steps, completed_jobs, metrics)
    """
    warehouse, robot_count, order_probability, max_steps = params
    steps, jobs = run_simulation_until_complete(
        warehouse, 
        robot_count,
        order_probability,
        max_steps
    )
    
    metrics = {
        'robot_count': robot_count,
        'order_probability': order_probability,
        'steps': steps,
        'completed_jobs': jobs,
        'is_timeout': steps >= max_steps
    }
    
    return steps, jobs, metrics

def generate_training_data(
    num_samples: int = 100,
    warehouse_sizes: List[Tuple[int, int]] = [(10, 10), (15, 15), (20, 20)],
    robot_ratios: List[float] = [0.2, 0.4, 0.6, 0.8],
    workstation_ratios: List[float] = [0.1, 0.2, 0.3],
    order_densities: List[float] = [0.2, 0.4, 0.6],
    max_steps: int = 5000,
    n_processes: Optional[int] = None
) -> pd.DataFrame:
    """Generate training data using parallel processing
    
    Args:
        num_samples: Number of samples per configuration
        warehouse_sizes: List of warehouse dimensions to test
        robot_ratios: List of robot count ratios (relative to warehouse size)
        workstation_ratios: List of workstation count ratios
        order_densities: List of order density probabilities
        max_steps: Maximum simulation steps
        n_processes: Number of processes to use (defaults to CPU count - 1)
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"Generating training data using {n_processes} processes...")
    
    all_results = []
    
    # Create a pool of worker processes
    with mp.Pool(processes=n_processes) as pool:
        # For each warehouse size
        for size in warehouse_sizes:
            rows, cols = size
            print(f"\nProcessing warehouse size {size}")
            
            # Generate warehouse layout
            warehouse = generate_warehouse_layout(rows, cols)
            available_space = np.sum(warehouse == 0)
            
            # Calculate robot and workstation counts
            robot_counts = [max(1, int(available_space * ratio)) for ratio in robot_ratios]
            workstation_counts = [max(1, int(available_space * ratio)) for ratio in workstation_ratios]
            
            # Generate all parameter combinations
            param_combinations = []
            for robot_count in robot_counts:
                for order_density in order_densities:
                    # Create num_samples copies of this configuration
                    for _ in range(num_samples):
                        param_combinations.append((
                            warehouse.copy(),
                            robot_count,
                            order_density,
                            max_steps
                        ))
            
            # Run simulations in parallel
            print(f"Running {len(param_combinations)} simulations...")
            results = list(tqdm.tqdm(
                pool.imap(run_parallel_simulations, param_combinations),
                total=len(param_combinations)
            ))
            
            # Process results
            for _, _, metrics in results:
                metrics['warehouse_size'] = rows * cols
                metrics['warehouse_rows'] = rows
                metrics['warehouse_cols'] = cols
                all_results.append(metrics)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate additional features
    df['robot_density'] = df['robot_count'] / df['warehouse_size']
    df['completion_rate'] = df['completed_jobs'] / (df['warehouse_size'] * df['order_probability'])
    df['steps_per_job'] = df['steps'] / df['completed_jobs']
    
    print("\nTraining data generation complete!")
    print(f"Generated {len(df)} samples")
    print("\nSample statistics:")
    print(df.describe())
    
    return df

def train_completion_time_model(data: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict]:
    """Train a RandomForest model to predict completion time
    
    Args:
        data: DataFrame containing simulation results
        
    Returns:
        Tuple of (trained model, feature importance dict)
    """
    # Prepare features
    features = [
        'warehouse_size', 'warehouse_rows', 'warehouse_cols',
        'robot_count', 'robot_density', 'order_probability'
    ]
    
    X = data[features]
    y = data['steps']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,  # Use all available cores
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\nModel Performance:")
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    print(f"Root Mean Squared Error: {rmse:.2f} steps")
    
    # Get feature importance
    importance = dict(zip(features, model.feature_importances_))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance in Completion Time Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    
    return model, importance

def validate_model_predictions(
    model: RandomForestRegressor,
    num_validation_samples: int = 20,
    warehouse_sizes: List[Tuple[int, int]] = [(25, 25), (30, 30)],  # Different from training
    max_steps: int = 5000,
    n_processes: Optional[int] = None
) -> None:
    """Validate model predictions on new warehouse configurations
    
    Args:
        model: Trained RandomForest model
        num_validation_samples: Number of validation samples per configuration
        warehouse_sizes: List of warehouse sizes to validate on
        max_steps: Maximum simulation steps
        n_processes: Number of processes to use (defaults to CPU count - 1)
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"\nValidating model predictions using {n_processes} processes...")
    
    validation_results = []
    
    # Create a pool of worker processes
    with mp.Pool(processes=n_processes) as pool:
        for size in warehouse_sizes:
            rows, cols = size
            print(f"\nValidating warehouse size {size}")
            
            # Generate warehouse layout
            warehouse = generate_warehouse_layout(rows, cols)
            available_space = np.sum(warehouse == 0)
            
            # Test different robot counts and order densities
            robot_counts = [int(available_space * ratio) for ratio in [0.3, 0.5, 0.7]]
            order_densities = [0.3, 0.5]
            
            # Generate parameter combinations
            param_combinations = []
            for robot_count in robot_counts:
                for order_density in order_densities:
                    for _ in range(num_validation_samples):
                        param_combinations.append((
                            warehouse.copy(),
                            robot_count,
                            order_density,
                            max_steps
                        ))
            
            # Run validation simulations in parallel
            print(f"Running {len(param_combinations)} validation simulations...")
            results = list(tqdm.tqdm(
                pool.imap(run_parallel_simulations, param_combinations),
                total=len(param_combinations)
            ))
            
            # Process results and make predictions
            for _, _, metrics in results:
                # Prepare features for prediction
                features = pd.DataFrame([{
                    'warehouse_size': rows * cols,
                    'warehouse_rows': rows,
                    'warehouse_cols': cols,
                    'robot_count': metrics['robot_count'],
                    'robot_density': metrics['robot_count'] / (rows * cols),
                    'order_probability': metrics['order_probability']
                }])
                
                # Make prediction
                predicted_steps = model.predict(features)[0]
                
                # Store results
                validation_results.append({
                    'actual_steps': metrics['steps'],
                    'predicted_steps': predicted_steps,
                    'error': predicted_steps - metrics['steps'],
                    'robot_count': metrics['robot_count'],
                    'order_probability': metrics['order_probability'],
                    'warehouse_size': rows * cols
                })
    
    # Convert to DataFrame
    validation_df = pd.DataFrame(validation_results)
    
    # Calculate error metrics
    mse = mean_squared_error(validation_df['actual_steps'], validation_df['predicted_steps'])
    rmse = np.sqrt(mse)
    r2 = r2_score(validation_df['actual_steps'], validation_df['predicted_steps'])
    
    print("\nValidation Results:")
    print(f"Root Mean Squared Error: {rmse:.2f} steps")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(validation_df['actual_steps'], validation_df['predicted_steps'], alpha=0.5)
    plt.plot([0, max(validation_df['actual_steps'])], [0, max(validation_df['actual_steps'])], 'r--')
    plt.xlabel('Actual Steps')
    plt.ylabel('Predicted Steps')
    plt.title('Model Predictions vs Actual Completion Times')
    plt.tight_layout()
    plt.savefig('prediction_validation.png', dpi=300, bbox_inches='tight')
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(validation_df['error'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Prediction Error (steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_errors.png', dpi=300, bbox_inches='tight')

def generate_warehouse_layout(rows: int, cols: int) -> np.ndarray:
    """Generate a warehouse layout with the specified dimensions"""
    # Simple layout generation - can be enhanced
    warehouse = np.ones((rows, cols))  # All shelves initially
    
    # Create aisles
    for i in range(1, rows-1, 3):
        warehouse[i, :] = 0  # Horizontal aisle
    for j in range(1, cols-1, 3):
        warehouse[:, j] = 0  # Vertical aisle
        
    return warehouse

if __name__ == "__main__":
    # Choose which analysis to run
    analysis_type = 4  # 4 = ML prediction model
    
    if analysis_type == 1:
        print("Running completion time distribution analysis...")
        analyze_completion_time_distribution(num_samples=1000)
    elif analysis_type == 2:
        print("Running robot count impact analysis...")
        analyze_robot_count_impact(num_samples=50)
    elif analysis_type == 3:
        print("Running directional aisle comparison...")
        compare_directional_aisles(num_samples=30)
    elif analysis_type == 4:
        print("Running machine learning analysis...")
        
        # Generate training data
        print("\nGenerating training data...")
        training_data = generate_training_data(
            num_samples=50,  # Reduced for testing, increase for better results
            warehouse_sizes=[(10, 10), (15, 15), (20, 20)],
            robot_ratios=[0.2, 0.4, 0.6],
            workstation_ratios=[0.1, 0.2],
            order_densities=[0.2, 0.4]
        )
        
        # Train model
        print("\nTraining model...")
        model, feature_importance = train_completion_time_model(training_data)
        
        # Validate model
        print("\nValidating model...")
        validate_model_predictions(
            model,
            num_validation_samples=20,
            warehouse_sizes=[(25, 25), (30, 30)]  # Test on larger warehouses
        ) 