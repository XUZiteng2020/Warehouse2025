import numpy as np
import pandas as pd
import os
import shutil
from generate_training_data import generate_dataset

# Constants
LARGE_WAREHOUSE_DIR = 'large_warehouse_data_files'

def generate_large_warehouse_data():
    """Generate dataset for large warehouses"""
    print("Generating large warehouse dataset...")
    
    # Create directory for large warehouse data
    if os.path.exists(LARGE_WAREHOUSE_DIR):
        print(f"Removing existing {LARGE_WAREHOUSE_DIR}...")
        shutil.rmtree(LARGE_WAREHOUSE_DIR)
    os.makedirs(LARGE_WAREHOUSE_DIR)
    
    # Generate data in the large warehouse directory
    return generate_dataset(
        warehouse_sizes=[(60, 100), (90, 150), (120, 200)],
        robot_ratios=[0.01, 0.02, 0.03, 0.05],
        workstation_counts=[2, 3, 4, 5, 6],
        order_densities=[0.05, 0.1, 0.2, 0.3],
        samples_per_config=1,
        max_steps=20000,
        save_dir=LARGE_WAREHOUSE_DIR,
        save_layouts=False  # Don't save layout files
    )

if __name__ == "__main__":
    data = generate_large_warehouse_data()
    print(f"\nLarge warehouse dataset generated and saved to {LARGE_WAREHOUSE_DIR}/")
    print(f"- Training data: {LARGE_WAREHOUSE_DIR}/training_data.csv") 