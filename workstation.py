import numpy as np
from typing import List, Tuple

def generate_workstation_positions(num_workstations: int, warehouse_height: int) -> List[Tuple[int, int]]:
    """
    Randomly place workstations along the left edge of the warehouse.
    
    Args:
        num_workstations: Number of workstations to place
        warehouse_height: Height of the warehouse
        
    Returns:
        List[Tuple[int, int]]: List of (row, col) positions for workstations
        
    Note:
        - Workstations are placed at column 0 (left edge)
        - Workstations are placed at random rows, avoiding the top 2 rows
        - No two workstations can occupy the same position
    """
    # Available rows for workstations (excluding top 2 rows)
    available_rows = list(range(2, warehouse_height))
    
    # Ensure we don't try to place more workstations than available positions
    num_workstations = min(num_workstations, len(available_rows))
    
    # Randomly select rows for workstations
    selected_rows = np.random.choice(
        available_rows, 
        size=num_workstations, 
        replace=False  # Ensure no duplicates
    )
    
    # Create list of workstation positions (row, col)
    workstation_positions = [(int(row), 0) for row in sorted(selected_rows)]
    
    return workstation_positions 