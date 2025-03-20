import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
from warehouse_manager import WarehouseManager
from order_generation import uniform_order_distribution
from multiprocessing import Pool, cpu_count
import itertools

def generate_warehouse_layout(width: int, height: int) -> np.ndarray:
    """
    Generate a warehouse layout with the following pattern:
    - 4 empty columns at the start
    - 5x2 shelf blocks
    - 2-cell wide aisles between shelf blocks
    - 2 empty rows at the top
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
        row += shelf_block_height + aisle_width
    
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

def generate_additional_data(save_path: str = 'warehouse_data_files'):
    """Generate additional training data for 50x85 warehouse focusing on 30-150 robots"""
    
    # Configuration
    width, height = 50, 85
    robot_counts = list(range(30, 151, 5))  # 30 to 150 robots in steps of 5
    workstation_counts = [2]  # Only 2 workstations
    target_orders = 1000
    samples_per_config = 5  # Increased samples per configuration for better statistics
    max_steps = 20000
    
    # Generate warehouse layout
    warehouse = generate_warehouse_layout(width, height)
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
            desc="Generating additional data"
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
    
    # Save results
    csv_path = os.path.join(save_path, 'training_data.csv')
    if os.path.exists(csv_path):
        # If file exists, append with new sample IDs
        existing_data = pd.read_csv(csv_path)
        max_sample_id = existing_data['sample_id'].max() if 'sample_id' in existing_data.columns else -1
        results_df['sample_id'] += int(max_sample_id + 1)
        results_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # If file doesn't exist, create new
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'layouts'), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
    
    # Save layout visualizations
    for i, layout in enumerate(all_layouts):
        layout_filename = f'layout_{int(results_df.iloc[i]["sample_id"]):04d}.npy'
        np.save(os.path.join(save_path, 'layouts', layout_filename), layout)
    
    return results_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate additional data
    print("Generating additional training data for 50x85 warehouse...")
    print("Focus: 30-150 robots, 2 workstations, 1000 orders")
    new_data = generate_additional_data()
    
    # Print summary of new data
    print("\nNew data generated:")
    print(f"Number of new samples: {len(new_data)}")
    
    # Calculate average metrics per robot count
    metrics_by_robots = new_data.groupby('num_robots').agg({
        'completion_time': ['mean', 'std'],
        'orders_per_step': ['mean', 'std'],
        'robot_efficiency': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance metrics by robot count:")
    print(metrics_by_robots)
    
    print("\nData saved to warehouse_data_files/training_data.csv") 