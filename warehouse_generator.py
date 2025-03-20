import numpy as np
import os
import glob
import pandas as pd

def generate_warehouse(rows, cols):
    """Generate a warehouse matrix based on the provided rules"""
    warehouse = np.zeros((rows, cols), dtype=int)
    
    # Starting column for shelves (after the initial four empty columns)
    start_col = 4
    shelf_width = 5
    aisle_width = 2
    shelf_height = 2
    
    # Place shelves in a grid-like pattern, following the given rules
    row_idx = 0
    while row_idx + shelf_height <= rows:
        col_idx = start_col
        while col_idx + shelf_width <= cols:
            # Fill a shelf
            warehouse[row_idx:row_idx + shelf_height, col_idx:col_idx + shelf_width] = 1
            # Move over by the shelf width plus aisle width
            col_idx += shelf_width + aisle_width
        # Move down by the shelf height plus aisle width
        row_idx += shelf_height + aisle_width
    
    return warehouse

def save_warehouse_layout(layout, filename, layout_type="original"):
    """Save warehouse layout to file with metadata"""
    # Create layouts directory if it doesn't exist
    layouts_dir = 'layouts'
    os.makedirs(layouts_dir, exist_ok=True)
    
    # Add metadata as header
    metadata = {
        'type': layout_type,
        'height': layout.shape[0],
        'width': layout.shape[1],
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Save layout with metadata
    filepath = os.path.join(layouts_dir, filename)
    with open(filepath, 'w') as f:
        # Write metadata
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        # Write layout data
        np.savetxt(f, layout, fmt='%d')
    
    print(f"Layout saved to {filepath}")
    return filepath

def main():
    # Generate warehouse layouts
    rows, cols = 50, 85  # Standard size
    
    # Generate original layout
    warehouse_original = generate_warehouse(rows, cols)
    # Add two rows of aisles on top
    warehouse_original = np.vstack((np.zeros((2, cols), dtype=int), warehouse_original))
    
    # Save original layout
    save_warehouse_layout(warehouse_original, 'warehouse_original.txt', 'original')
    
    print("Warehouse layouts generated successfully.")

if __name__ == "__main__":
    main()
