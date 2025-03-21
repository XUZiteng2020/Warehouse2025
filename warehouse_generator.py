import numpy as np
import os
import glob

# Function to generate a warehouse matrix based on the provided rules
def generate_warehouse(rows, cols):
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

# Generate a larger warehouse (e.g., 12 rows by 20 columns)
rows, cols = 30,50
warehouse_data = generate_warehouse(rows, cols)

# Add two rows of aisles on top by prepending rows of zeros
warehouse_data_with_aisles = np.vstack((np.zeros((2, cols), dtype=int), warehouse_data))
# Directory containing the warehouse data files
data_files_dir = '/Users/xuziteng/Documents/GitHub/Warehouse2025/warehouse_data_files/'

# Delete all files in the warehouse_data_files directory
files = glob.glob(os.path.join(data_files_dir, '*'))
for f in files:
    os.remove(f)

# Save the final warehouse data to a file
np.savetxt(os.path.join(data_files_dir, 'warehouse_data.txt'), warehouse_data_with_aisles, fmt='%d')
