import numpy as np
import os
import pandas as pd

def generate_original_layout(rows, cols):
    """Generate original warehouse layout with bi-directional aisles"""
    warehouse = np.zeros((rows, cols), dtype=int)
    
    # Configuration
    start_col = 4  # Left margin
    shelf_width = 5  # Width of each shelf block
    aisle_width = 2  # Width of normal aisles
    shelf_height = 2  # Height of each shelf block
    
    # Place shelves in a grid pattern
    row = 0
    while row + shelf_height <= rows:
        col = start_col
        while col + shelf_width <= cols:
            # Place shelf block
            warehouse[row:row + shelf_height, col:col + shelf_width] = 1
            col += shelf_width + aisle_width
        row += shelf_height + aisle_width
    
    return warehouse

def generate_two_main_roads_layout(rows, cols):
    """
    Generate warehouse layout with two horizontal main roads and normal aisles
    Layout encoding:
     0: Empty space (for general movement)
     1: Storage shelf
    -1: Main road (horizontal only)
    -2: Normal aisle
    """
    warehouse = np.zeros((rows, cols), dtype=int)
    
    # Configuration
    start_col = 4  # Left margin
    shelf_width = 5  # Width of each shelf block
    aisle_width = 2  # Width of normal aisles
    main_road_width = 3  # Width of main roads
    shelf_height = 2  # Height of each shelf block
    
    # Calculate main road positions - at 1/3 and 2/3 of height
    horizontal_main_roads = [
        rows // 3,
        2 * rows // 3
    ]
    
    # Place shelves and normal aisles
    row = 0
    while row + shelf_height <= rows:
        # Skip rows where main roads go
        if any(hmr <= row < hmr + main_road_width for hmr in horizontal_main_roads):
            # Place horizontal main road
            warehouse[row:row + main_road_width, start_col:] = -1
            row += main_road_width
            continue
            
        col = start_col
        while col + shelf_width <= cols:
            # Place shelf block
            warehouse[row:row + shelf_height, col:col + shelf_width] = 1
            
            # Place normal aisle after shelf
            if col + shelf_width + aisle_width <= cols:
                next_col = col + shelf_width
                warehouse[row:row + shelf_height, next_col:next_col + aisle_width] = -2
            
            col += shelf_width + aisle_width
        row += shelf_height + aisle_width
    
    return warehouse

def visualize_layout(layout):
    """Create a string visualization of the warehouse layout"""
    symbols = {
        0: "  ",   # Empty space
        1: "██",   # Shelf
        -1: "══",  # Main road
        -2: "||"   # Normal aisle
    }
    
    rows, cols = layout.shape
    visualization = "Warehouse Layout Legend:\n"
    visualization += "██ : Storage Shelf\n"
    visualization += "══ : Main Road (Horizontal)\n"
    visualization += "|| : Normal Aisle\n"
    visualization += "   : Empty Space\n\n"
    
    # Add column numbers
    visualization += "   "
    for col in range(cols):
        visualization += f"{col%10:2}"
    visualization += "\n"
    
    # Add the layout
    for row in range(rows):
        visualization += f"{row:2} "
        for col in range(cols):
            visualization += symbols[layout[row, col]]
        visualization += "\n"
    
    return visualization

def save_layout_visualization(layout, filename):
    """Save the layout visualization to a text file"""
    vis = visualize_layout(layout)
    with open(filename, 'w') as f:
        f.write(vis)
    print(f"Layout visualization saved to {filename}")

def main():
    """Generate and compare both layout designs"""
    # Create example layouts with dimensions matching simulation size
    rows, cols = 50, 85  # Changed from 20, 40 to match simulation dimensions
    
    # Generate both layouts
    original = generate_original_layout(rows, cols)
    two_main_roads = generate_two_main_roads_layout(rows, cols)
    
    # Create layouts directory if it doesn't exist
    os.makedirs('layouts', exist_ok=True)
    
    # Save and display original layout
    np.savetxt('layouts/original_layout.txt', original, fmt='%d')
    save_layout_visualization(original, 'layouts/original_layout_visual.txt')
    print("\nOriginal Layout Design (bi-directional aisles):")
    print(visualize_layout(original))
    
    # Save and display two main roads layout
    np.savetxt('layouts/two_main_roads_layout.txt', two_main_roads, fmt='%d')
    save_layout_visualization(two_main_roads, 'layouts/two_main_roads_layout_visual.txt')
    print("\nTwo Main Roads Layout Design:")
    print(visualize_layout(two_main_roads))
    
    # Calculate and display space efficiency
    def calculate_efficiency(layout):
        total_cells = layout.size
        shelf_cells = np.sum(layout == 1)
        road_cells = np.sum((layout == -1) | (layout == -2))
        empty_cells = np.sum(layout == 0)
        
        # Calculate actual dimensions in cells
        rows, cols = layout.shape
        dimensions_str = f"Warehouse Dimensions: {rows}x{cols} cells"
        
        return {
            'total_cells': total_cells,
            'shelf_cells': shelf_cells,
            'road_cells': road_cells,
            'empty_cells': empty_cells,
            'storage_ratio': shelf_cells / total_cells,
            'road_ratio': road_cells / total_cells,
            'dimensions': dimensions_str
        }
    
    orig_eff = calculate_efficiency(original)
    two_main_eff = calculate_efficiency(two_main_roads)
    
    print("\nSpace Efficiency Comparison:")
    print(f"\n{orig_eff['dimensions']}")
    print(f"Original Layout:")
    print(f"- Storage space: {orig_eff['shelf_cells']} cells ({orig_eff['storage_ratio']:.2%})")
    print(f"- Road space: {orig_eff['road_cells']} cells ({orig_eff['road_ratio']:.2%})")
    print(f"- Empty space: {orig_eff['empty_cells']} cells ({orig_eff['empty_cells']/orig_eff['total_cells']:.2%})")
    
    print(f"\n{two_main_eff['dimensions']}")
    print(f"Two Main Roads Layout:")
    print(f"- Storage space: {two_main_eff['shelf_cells']} cells ({two_main_eff['storage_ratio']:.2%})")
    print(f"- Road space: {two_main_eff['road_cells']} cells ({two_main_eff['road_ratio']:.2%})")
    print(f"- Empty space: {two_main_eff['empty_cells']} cells ({two_main_eff['empty_cells']/two_main_eff['total_cells']:.2%})")
    
    # Print shelf and road dimensions for reference
    print("\nLayout Configuration:")
    print("- Shelf size: 5x2 cells")
    print("- Normal aisle width: 2 cells")
    print("- Main road width: 3 cells")
    print("- Left margin: 4 cells")

if __name__ == "__main__":
    main()
