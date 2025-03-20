import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from generate_training_data import generate_warehouse_layout, run_simulation

def generate_large_warehouse_data():
    """Generate dataset for large warehouses"""
    print("Generating large warehouse dataset...")
    
    # Define configurations for large warehouses
    warehouse_sizes = [(60, 100), (90, 150), (120, 200)]
    robot_ratios = [0.03, 0.05, 0.07, 0.1]  # Percentage of shelf count
    workstation_counts = [2, 3, 4, 5, 6]
    order_densities = [0.05, 0.1, 0.2, 0.3]
    samples_per_config = 1
    max_steps = 20000
    
    # Create save directory if it doesn't exist
    save_path = 'warehouse_data_files'
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize results
    all_results = []
    sample_id = 0
    csv_path = os.path.join(save_path, 'large_warehouse_data.csv')
    
    # Calculate total combinations
    total_combinations = (len(warehouse_sizes) * len(robot_ratios) * 
                        len(workstation_counts) * len(order_densities) * 
                        samples_per_config)
    
    with tqdm(total=total_combinations, desc="Generating large warehouse dataset") as pbar:
        for width, height in warehouse_sizes:
            warehouse = generate_warehouse_layout(width, height)
            shelf_count = np.sum(warehouse == 1)
            
            for robot_ratio in robot_ratios:
                num_robots = max(1, int(shelf_count * robot_ratio))
                
                for num_workstations in workstation_counts:
                    for order_density in order_densities:
                        for sample in range(samples_per_config):
                            # Run simulation
                            result, _ = run_simulation(
                                warehouse=warehouse,
                                num_robots=num_robots,
                                num_workstations=num_workstations,
                                order_density=order_density,
                                max_steps=max_steps
                            )
                            
                            # Add shelf count and robot ratio to results
                            result['shelf_count'] = shelf_count
                            result['robot_ratio'] = robot_ratio
                            result['sample_id'] = sample_id
                            
                            # Save result immediately
                            all_results.append(result)
                            df_current = pd.DataFrame([result])
                            if os.path.exists(csv_path):
                                df_current.to_csv(csv_path, mode='a', header=False, index=False)
                            else:
                                df_current.to_csv(csv_path, mode='w', header=True, index=False)
                            
                            sample_id += 1
                            pbar.update(1)
    
    print(f"\nLarge warehouse dataset generated and saved to {csv_path}")
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    generate_large_warehouse_data() 