import pygame
import numpy as np
import os
import random
from order_generation import uniform_order_distribution

warehouse_dir = 'warehouse_data_files'
warehouse_files = [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')]
random_warehouse_file = os.path.join(warehouse_dir, random.choice(warehouse_files))
warehouse = np.loadtxt(random_warehouse_file)

def main():
    pygame.init()
    
    # Define colors (RGB)
    BLACK = (0, 0, 0)       # Aisles background
    WHITE = (255, 255, 255) # Shelves background and button background
    RED   = (255, 0, 0)     # Order indicator
    GRAY  = (200, 200, 200) # Grid lines and background for extra space
    
    # Set cell size for drawing the warehouse grid
    cell_size = 40  # You can adjust the size as needed
    rows, cols = warehouse.shape
    grid_width = cols * cell_size
    grid_height = rows * cell_size
    
    # Define button dimensions (displayed to the right of the grid)
    button_width = 200
    button_height = 50
    margin = 20  # margin around button and grid
    
    # Window dimensions: grid + button + margins
    window_width = grid_width + button_width + 3 * margin
    window_height = max(grid_height + 2 * margin, button_height + 2 * margin)
    
    # Set up the Pygame window
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Warehouse Order Generation")
    
    # Define the button rectangle (positioned on the right)
    button_rect = pygame.Rect(grid_width + 2 * margin, margin, button_width, button_height)
    
    # Load a basic font
    font = pygame.font.SysFont(None, 30)
    
    # Initially, no orders are generated; set orders array to zeros
    orders = np.zeros_like(warehouse)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Check for mouse click events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos):
                    # When button is clicked, generate orders with probability p (e.g., 0.3)
                    orders = uniform_order_distribution(0.3)
        
        # Fill the background
        screen.fill(GRAY)
        
        # Draw the warehouse grid
        for i in range(rows):
            for j in range(cols):
                # Compute the rectangle for each cell
                cell_rect = pygame.Rect(j * cell_size + margin,
                                        i * cell_size + margin,
                                        cell_size, cell_size)
                # Determine base color from the warehouse layout:
                if warehouse[i, j] == 0:
                    color = BLACK  # aisle
                else:
                    color = WHITE  # shelf
                
                pygame.draw.rect(screen, color, cell_rect)
                
                # If the cell is a shelf and an order is present, overlay red
                if warehouse[i, j] == 1 and orders[i, j] == 1:
                    pygame.draw.rect(screen, RED, cell_rect)
                
                # Optional: draw cell border for clarity
                pygame.draw.rect(screen, GRAY, cell_rect, 1)
        
        # Draw the "Order Generation" button
        pygame.draw.rect(screen, WHITE, button_rect)
        pygame.draw.rect(screen, BLACK, button_rect, 2)  # border
        
        # Render button text
        button_text = font.render("Order Generation", True, BLACK)
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == '__main__':
    main()