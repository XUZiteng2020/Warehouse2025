import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import shutil
from warehouse_manager import WarehouseManager
from order_generation import uniform_order_distribution

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

def run_simulation(warehouse: np.ndarray, 
                  num_robots: int,
                  num_workstations: int,
                  order_density: float,
                  max_steps: int = 5000) -> dict:
    """Run a single simulation and return metrics"""
    
    # Initialize warehouse manager
    manager = WarehouseManager(
        warehouse=warehouse,
        num_workstations=num_workstations,
        collision_method=1  # Using waiting method
    )
    manager.initialize_robots(num_robots)
    
    # Generate initial orders
    orders = uniform_order_distribution(order_density, warehouse)
    manager.update_orders(orders)
    
    # Create layout visualization (0=empty, 1=shelf, 2=workstation, 3=order)
    layout_viz = warehouse.copy()
    # Add workstations
    for wx, wy in manager.workstations:
        layout_viz[wx, wy] = 2
    # Add orders
    layout_viz[orders == 1] = 3
    
    # Run simulation
    total_orders = np.sum(orders)
    step = 0
    manager.toggle_play()  # Start simulation
    
    # Run until either max steps reached or all orders completed
    while step < max_steps:
        manager.update()
        step += 1
        
        # Check if all orders are completed
        if np.sum(manager.orders) == 0 and len(manager.targeted_orders) == 0:
            break
    
    # Collect metrics
    result = {
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
    }
    
    return result, layout_viz

def generate_dataset(
        warehouse_sizes=[(30, 50), (40, 70), (50, 85)],
        robot_ratios=[0.01, 0.02, 0.03, 0.05],
        workstation_counts=[2, 3, 4, 5, 6],
        order_densities=[0.05, 0.1, 0.2, 0.3],
        samples_per_config=3,
        max_steps=20000,
        save_dir='warehouse_data_files',
        save_layouts=True
    ):
    """Generate a dataset of warehouse simulations with various configurations"""
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    if save_layouts:
        os.makedirs(os.path.join(save_dir, 'layouts'), exist_ok=True)
    
    # Initialize lists to store data
    data = []
    layout_idx = 0
    
    # Generate data for each configuration
    total_configs = len(warehouse_sizes) * len(robot_ratios) * \
                   len(workstation_counts) * len(order_densities) * samples_per_config
    
    with tqdm(total=total_configs, desc="Generating dataset") as pbar:
        for w_size in warehouse_sizes:
            width, height = w_size
            for robot_ratio in robot_ratios:
                num_robots = int(width * height * robot_ratio)
                for ws_count in workstation_counts:
                    for order_density in order_densities:
                        for _ in range(samples_per_config):
                            # Generate warehouse layout
                            layout = generate_warehouse_layout(width, height)
                            
                            # Save layout if requested
                            if save_layouts:
                                layout_file = f'layout_{layout_idx:04d}.npy'
                                np.save(os.path.join(save_dir, 'layouts', layout_file), layout)
                            
                            # Run simulation
                            shelf_count = np.sum(layout == 2)  # Count shelf cells
                            results, layout_viz = run_simulation(
                                warehouse=layout,
                                num_robots=num_robots,
                                num_workstations=ws_count,
                                order_density=order_density,
                                max_steps=max_steps
                            )
                            
                            # Store data
                            data.append({
                                'warehouse_width': width,
                                'warehouse_height': height,
                                'num_robots': num_robots,
                                'num_workstations': ws_count,
                                'order_density': order_density,
                                'shelf_count': shelf_count,
                                'completion_time': results['completion_time'],
                                'orders_per_step': results['orders_per_step'],
                                'robot_efficiency': results['robot_efficiency'],
                                'completion_rate': results['completion_rate']
                            })
                            
                            layout_idx += 1
                            pbar.update(1)
    
    # Save data to CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'training_data.csv'), index=False)
    
    return df

if __name__ == "__main__":
    generate_dataset(
        warehouse_sizes=[(30, 50), (40, 70), (50, 85)],
        robot_ratios=[0.01, 0.02, 0.03, 0.05],
        workstation_counts=[2, 3, 4, 5, 6],
        order_densities=[0.05, 0.1, 0.2, 0.3],
        samples_per_config=3,
        max_steps=20000
    )