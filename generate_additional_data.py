import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
from warehouse_manager import WarehouseManager
from order_generation import uniform_order_distribution
from multiprocessing import Pool, cpu_count
import itertools
from warehouse_simulator import WarehouseSimulator
from warehouse_generator import generate_original_layout, generate_two_main_roads_layout
import multiprocessing as mp

def generate_original_layout(width: int, height: int) -> np.ndarray:
    """
    Generate a warehouse layout with the original bi-directional aisles pattern:
    - 4 empty columns at the start
    - 5x2 shelf blocks
    - 2-cell wide aisles between shelf blocks
    - 2 empty rows at the top
    - Each horizontal aisle pair has opposite directions
    """
    layout = np.zeros((height, width), dtype=int)
    
    # Start placing shelves after initial empty space
    start_row = 2  # 2 empty rows at top
    start_col = 4  # 4 empty columns at start
    
    # Place shelves in a grid pattern
    shelf_block_width = 5
    shelf_block_height = 2
    aisle_width = 2
    
    row = start_row
    while row + shelf_block_height <= height:
        col = start_col
        while col + shelf_block_width <= width:
            layout[row:row + shelf_block_height, col:col + shelf_block_width] = 1
            col += shelf_block_width + aisle_width
        
        # Mark aisle directions for each pair (one lane right, one lane left)
        if row > start_row:  # If this is an aisle (not the first row)
            # Upper lane goes right (-1), lower lane goes left (-2)
            layout[row - aisle_width, start_col:] = -1  # Upper lane rightward
            layout[row - aisle_width + 1, start_col:] = -2  # Lower lane leftward
            
        row += shelf_block_height + aisle_width
    
    return layout

def generate_warehouse_layout(width: int, height: int, layout_type: str = "main_road") -> np.ndarray:
    """
    Generate a warehouse layout with the specified pattern.
    
    Args:
        width: Warehouse width
        height: Warehouse height
        layout_type: "main_road" for single-direction aisles or "original" for bi-directional pairs
    """
    if layout_type == "original":
        return generate_original_layout(width, height)
    else:  # main_road
        layout = np.zeros((height, width), dtype=int)
        
        # Configuration parameters
        start_row = 2  # 2 empty rows at top
        start_col = 4  # 4 empty columns at start
        shelf_block_width = 5
        shelf_block_height = 2
        main_road_width = 3  # Increased from 2 to 3
        intersection_buffer = 1  # Buffer zone at intersections
        
        # Keep track of aisle directions (0: rightward, 2: leftward)
        aisle_directions = []
        current_direction = 0  # Start with rightward
        
        row = start_row
        aisle_count = 0
        while row + shelf_block_height <= height:
            col = start_col
            while col + shelf_block_width <= width:
                # Add buffer zone before shelf block at intersections
                if col > start_col:
                    layout[row:row + shelf_block_height, col - intersection_buffer:col] = -3  # Buffer zone
                
                # Place shelf block
                layout[row:row + shelf_block_height, col:col + shelf_block_width] = 1
                col += shelf_block_width + main_road_width
            
            # Store direction for this aisle pair
            if row > start_row:  # If this is an aisle (not the first row)
                aisle_directions.append((row - main_road_width, current_direction))
                # Toggle direction for next aisle
                current_direction = 2 if current_direction == 0 else 0
                aisle_count += 1
                
            row += shelf_block_height + main_road_width
        
        # Mark main road directions
        # Vertical main roads (alternating directions)
        for col in range(start_col + shelf_block_width + intersection_buffer, width, shelf_block_width + main_road_width):
            if col + main_road_width <= width:
                # Mark vertical main road sections
                layout[start_row:, col:col + main_road_width] = -4  # Vertical main road
        
        # Horizontal main roads
        for row in range(start_row + shelf_block_height, height, shelf_block_height + main_road_width):
            if row + main_road_width <= height:
                # Mark horizontal main road sections
                layout[row:row + main_road_width, start_col:] = -5  # Horizontal main road
        
        # Mark intersection zones
        for row in range(start_row + shelf_block_height, height, shelf_block_height + main_road_width):
            for col in range(start_col + shelf_block_width, width, shelf_block_width + main_road_width):
                if row + main_road_width <= height and col + main_road_width <= width:
                    layout[row:row + main_road_width, col:col + main_road_width] = -6  # Intersection
        
        return layout

def run_single_simulation(params):
    """Run a single simulation with given parameters"""
    warehouse, num_robots, num_workstations, order_density, max_steps = params
    
    # Initialize warehouse manager
    manager = WarehouseManager(
        warehouse=warehouse,
        num_workstations=num_workstations,
        collision_method=1
    )
    manager.initialize_robots(num_robots)
    
    # Generate initial orders
    orders = uniform_order_distribution(order_density, warehouse)
    manager.update_orders(orders)
    
    # Create layout visualization
    layout_viz = warehouse.copy()
    for wx, wy in manager.workstations:
        layout_viz[wx, wy] = 2
    layout_viz[orders == 1] = 3
    
    # Run simulation
    total_orders = np.sum(orders)
    step = 0
    manager.toggle_play()
    
    while step < max_steps:
        manager.update()
        step += 1
        
        if np.sum(manager.orders) == 0 and len(manager.targeted_orders) == 0:
            break
    
    # Return results
    return {
        'warehouse_width': warehouse.shape[1],
        'warehouse_height': warehouse.shape[0],
        'num_robots': num_robots,
        'num_workstations': num_workstations,
        'order_density': order_density,
        'total_orders': total_orders,
        'completed_orders': manager.completed_jobs,
        'completion_time': step,
        'orders_per_step': manager.completed_jobs / step if step > 0 else 0,
        'robot_efficiency': manager.completed_jobs / (num_robots * step) if step > 0 else 0,
        'completion_rate': manager.completed_jobs / total_orders if total_orders > 0 else 0
    }, layout_viz

def generate_additional_data(save_path: str = 'warehouse_data_files', layout_type: str = "main_road"):
    """Generate additional training data for 50x85 warehouse focusing on 30-150 robots"""
    
    # Configuration
    width, height = 50, 85
    robot_counts = list(range(30, 151, 5))  # 30 to 150 robots in steps of 5
    workstation_counts = [2]  # Only 2 workstations
    target_orders = 1000
    samples_per_config = 5  # Increased samples per configuration for better statistics
    max_steps = 20000
    
    # Generate warehouse layout
    warehouse = generate_warehouse_layout(width, height, layout_type)
    shelf_count = np.sum(warehouse == 1)
    
    # Calculate order density for target orders
    order_density = target_orders / shelf_count
    
    # Prepare simulation parameters
    simulation_params = []
    for num_robots in robot_counts:
        for num_workstations in workstation_counts:
            for _ in range(samples_per_config):
                simulation_params.append((
                    warehouse.copy(),
                    num_robots,
                    num_workstations,
                    order_density,
                    max_steps
                ))
    
    # Set up multiprocessing
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Running simulations using {num_processes} processes...")
    print(f"Total configurations to test: {len(simulation_params)}")
    print(f"Robot range: {min(robot_counts)} to {max(robot_counts)} in steps of 5")
    
    # Run simulations in parallel
    all_results = []
    all_layouts = []
    
    with Pool(num_processes) as pool:
        for result, layout in tqdm(
            pool.imap_unordered(run_single_simulation, simulation_params),
            total=len(simulation_params),
            desc=f"Generating data for {layout_type} layout"
        ):
            all_results.append(result)
            all_layouts.append(layout)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Add robot ratio and sample ID
    results_df['robot_ratio'] = results_df['num_robots'] / shelf_count
    results_df['shelf_count'] = shelf_count
    results_df['sample_id'] = range(len(results_df))
    results_df['sample_id'] = results_df['sample_id'].astype(int)
    results_df['layout_type'] = layout_type
    
    # Save results
    csv_path = os.path.join(save_path, f'training_data_{layout_type}.csv')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'layouts'), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    
    # Save layout visualizations
    for i, layout in enumerate(all_layouts):
        layout_filename = f'layout_{layout_type}_{int(results_df.iloc[i]["sample_id"]):04d}.npy'
        np.save(os.path.join(save_path, 'layouts', layout_filename), layout)
    
    return results_df

def run_simulation(config):
    """Run a single simulation with given configuration"""
    layout_type, num_robots, num_workstations, num_orders = config
    
    # Generate layout based on type
    if layout_type == "original":
        layout = generate_original_layout(50, 85)
    else:  # two_main_roads
        layout = generate_two_main_roads_layout(50, 85)
    
    simulator = WarehouseSimulator(
        layout=layout,
        num_robots=num_robots,
        num_workstations=num_workstations,
        target_orders=num_orders
    )
    
    # Run simulation and get metrics
    completion_time, orders_per_step = simulator.run()
    robot_efficiency = orders_per_step / num_robots if num_robots > 0 else 0
    
    return {
        'layout_type': layout_type,
        'num_robots': num_robots,
        'num_workstations': num_workstations,
        'num_orders': num_orders,
        'completion_time': completion_time,
        'orders_per_step': orders_per_step,
        'robot_efficiency': robot_efficiency
    }

def main():
    """Generate simulation data for both layouts"""
    print("Generating additional training data for warehouse size 50x85...")
    
    # Configuration parameters
    robot_counts = list(range(30, 155, 5))  # 30 to 150 robots in steps of 5
    workstation_counts = [2]  # Fixed at 2 workstations
    order_counts = [1000]  # Fixed at 1000 orders
    layout_types = ["original", "two_main_roads"]
    samples_per_config = 5  # Multiple samples per configuration for statistical significance
    
    # Generate all configurations to test
    configs = []
    for layout_type in layout_types:
        for num_robots in robot_counts:
            for num_workstations in workstation_counts:
                for num_orders in order_counts:
                    for _ in range(samples_per_config):
                        configs.append((layout_type, num_robots, num_workstations, num_orders))
    
    print(f"Testing {len(configs)} configurations...")
    
    # Run simulations using multiple processes
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Running simulations using {num_processes} processes...")
    
    with mp.Pool(num_processes) as pool:
        results = []
        completed = 0
        
        # Run simulations and track progress
        for result in pool.imap_unordered(run_simulation, configs):
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.0f}%)")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('warehouse_data_files/training_data.csv', index=False)
    print(f"\nGenerated {len(df)} new samples")
    
    # Print summary statistics by layout type and robot count
    for layout_type in layout_types:
        print(f"\nPerformance metrics for {layout_type} layout:")
        layout_data = df[df['layout_type'] == layout_type]
        for num_robots in robot_counts:
            robot_data = layout_data[layout_data['num_robots'] == num_robots]
            if len(robot_data) > 0:
                print(f"\nRobots: {num_robots}")
                print(f"Completion Time: {robot_data['completion_time'].mean():.1f} ± {robot_data['completion_time'].std():.1f}")
                print(f"Orders/Step: {robot_data['orders_per_step'].mean():.3f} ± {robot_data['orders_per_step'].std():.3f}")
                print(f"Robot Efficiency: {robot_data['robot_efficiency'].mean():.3f} ± {robot_data['robot_efficiency'].std():.3f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data for both layouts
    print("Generating training data for 50x85 warehouse...")
    print("Focus: 30-150 robots, 2 workstations, 1000 orders")
    
    # Generate data for main road layout
    print("\nGenerating data for main road layout...")
    main_road_data = generate_additional_data(layout_type="main_road")
    
    # Generate data for original layout
    print("\nGenerating data for original layout...")
    original_data = generate_additional_data(layout_type="original")
    
    # Print summary of new data
    print("\nData generated:")
    print(f"Main road layout samples: {len(main_road_data)}")
    print(f"Original layout samples: {len(original_data)}")
    
    print("\nData saved to:")
    print("- warehouse_data_files/training_data_main_road.csv")
    print("- warehouse_data_files/training_data_original.csv")
    
    # Calculate average metrics per robot count
    metrics_by_robots = main_road_data.groupby('num_robots').agg({
        'completion_time': ['mean', 'std'],
        'orders_per_step': ['mean', 'std'],
        'robot_efficiency': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance metrics by robot count for main road layout:")
    print(metrics_by_robots)
    
    metrics_by_robots = original_data.groupby('num_robots').agg({
        'completion_time': ['mean', 'std'],
        'orders_per_step': ['mean', 'std'],
        'robot_efficiency': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance metrics by robot count for original layout:")
    print(metrics_by_robots)
    
    main() 