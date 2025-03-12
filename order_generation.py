import numpy as np

# Example global warehouse: replace this with your actual warehouse array
# For demonstration, here's a small warehouse array.
# Get a random warehouse file from warehouse_data_files directory
# Get a random warehouse file from warehouse_data_files directory


def uniform_order_distribution(p, warehouse):
    """
    Generates a uniform order distribution for the warehouse.
    
    For each shelf (warehouse entry of 1), there is a probability `p` that an order is present (1),
    and probability (1-p) that no order is present (0). Aisles (warehouse entry of 0) are always 0 in the orders output.
    
    Parameters:
        p (float): Probability between 0 and 1 that a shelf has an order.
        warehouse (numpy.ndarray): 2D array representing the warehouse layout
        
    Returns:
        numpy.ndarray: A 2D numpy array of the same shape as 'warehouse', containing orders.
    """
    # Create a random matrix with values in [0, 1) of the same shape as the warehouse
    random_matrix = np.random.rand(*warehouse.shape)
    
    # For shelves (warehouse == 1), place an order (1) if the random value is less than p; otherwise 0.
    orders = np.where(warehouse == 1, (random_matrix < p).astype(int), 0)
    
    return orders

