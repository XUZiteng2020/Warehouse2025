import pygame
import numpy as np
import os
import random
from order_generation import uniform_order_distribution
from warehouse_manager import WarehouseManager
from robot import RobotStatus
import math


# Initialize warehouse
warehouse_dir = 'warehouse_data_files'
warehouse_files = [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')]
random_warehouse_file = os.path.join(warehouse_dir, random.choice(warehouse_files))
warehouse = np.loadtxt(random_warehouse_file)

def draw_robot(screen, x, y, cell_size, margin, rotation, is_working):
    """Draw a robot with its orientation arrow and working status indicator"""
    # Convert warehouse coordinates to screen coordinates
    screen_x = y * cell_size + margin  # Swap x and y for proper visualization
    screen_y = x * cell_size + margin
    
    robot_size = int(cell_size * 0.7)
    robot_margin = (cell_size - robot_size) // 2
    
    # Robot body (orange square)
    robot_rect = pygame.Rect(
        screen_x + robot_margin,
        screen_y + robot_margin,
        robot_size, robot_size
    )
    pygame.draw.rect(screen, (255, 165, 0), robot_rect)  # Orange
    
    # Direction arrow
    center_x = screen_x + cell_size // 2
    center_y = screen_y + cell_size // 2
    arrow_length = robot_size // 2
    
    # Calculate arrow endpoint using rotation
    end_x = center_x + arrow_length * math.cos(math.radians(rotation))
    end_y = center_y - arrow_length * math.sin(math.radians(rotation))
    
    pygame.draw.line(screen, (0, 0, 0), (center_x, center_y), (end_x, end_y), 2)
    
    # Working status indicator (red dot)
    if is_working:
        dot_radius = robot_size // 6
        pygame.draw.circle(screen, (255, 0, 0),
                         (center_x, screen_y + robot_margin // 2),
                         dot_radius)

def main():
    pygame.init()
    
    # Define colors (RGB)
    BLACK = (0, 0, 0)       # Aisles background
    WHITE = (255, 255, 255) # Shelves background and button background
    RED = (255, 0, 0)      # Order indicator
    GREEN = (0, 255, 0)    # Workstation indicator
    GRAY = (200, 200, 200) # Grid lines and background
    
    # Initialize warehouse manager
    warehouse_manager = WarehouseManager(warehouse)
    print(f"Warehouse shape: {warehouse.shape}")
    print(f"Workstations: {warehouse_manager.workstations}")
    
    # Set cell size for drawing the warehouse grid
    cell_size = 40
    rows, cols = warehouse.shape
    grid_width = cols * cell_size
    grid_height = rows * cell_size
    
    # Define UI elements dimensions
    button_width = 200
    button_height = 50
    margin = 20
    slider_width = 200
    slider_height = 20
    
    # Window dimensions
    window_width = grid_width + button_width + 3 * margin
    window_height = max(grid_height + 2 * margin, 5 * (button_height + margin))
    
    # Set up the Pygame window
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Warehouse Simulation")
    
    # Define UI element rectangles
    order_button_rect = pygame.Rect(grid_width + 2 * margin, margin,
                                  button_width, button_height)
    play_button_rect = pygame.Rect(grid_width + 2 * margin, 2 * margin + button_height,
                                 button_width, button_height)
    evaluate_button_rect = pygame.Rect(grid_width + 2 * margin, 3 * margin + 2 * button_height,
                                     button_width, button_height)
    
    # Robot count slider
    slider_rect = pygame.Rect(grid_width + 2 * margin, 4 * margin + 3 * button_height,
                            slider_width, slider_height)
    max_robots = sum(1 for i in range(rows) for j in range(cols) 
                    if warehouse[i, j] == 0) - len(warehouse_manager.workstations)
    robot_count = max_robots // 2  # Start with half max robots
    warehouse_manager.initialize_robots(robot_count)
    print(f"Initialized {robot_count} robots")
    
    # Load font
    font = pygame.font.SysFont(None, 30)
    
    # Game loop
    clock = pygame.time.Clock()
    running = True
    dragging_slider = False
    frame_count = 0
    
    while running:
        frame_count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Order generation button
                if order_button_rect.collidepoint(mouse_pos):
                    orders = uniform_order_distribution(0.3, warehouse)
                    warehouse_manager.update_orders(orders)
                    print("Generated new orders")
                
                # Play/Pause button
                elif play_button_rect.collidepoint(mouse_pos):
                    warehouse_manager.toggle_play()
                
                # Evaluate button
                elif evaluate_button_rect.collidepoint(mouse_pos):
                    pass  # Handled in display
                
                # Slider
                elif slider_rect.collidepoint(mouse_pos):
                    dragging_slider = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False
            
            elif event.type == pygame.MOUSEMOTION and dragging_slider:
                mouse_x = pygame.mouse.get_pos()[0]
                relative_x = min(max(mouse_x - slider_rect.x, 0), slider_width)
                new_robot_count = int((relative_x / slider_width) * max_robots)
                if new_robot_count != robot_count:
                    robot_count = max(1, new_robot_count)
                    warehouse_manager.initialize_robots(robot_count)
                    print(f"Updated robot count to {robot_count}")
        
        # Update warehouse state
        if warehouse_manager.update() and frame_count % 60 == 0:
            print(f"Time step: {warehouse_manager.time_step}")
            for robot in warehouse_manager.robots:
                print(f"Robot at ({robot.x}, {robot.y}), status: {robot.status.value}")
        
        # Draw everything
        screen.fill(GRAY)
        
        # Draw warehouse grid
        for i in range(rows):
            for j in range(cols):
                cell_rect = pygame.Rect(j * cell_size + margin,
                                      i * cell_size + margin,
                                      cell_size, cell_size)
                
                # Base color (aisle or shelf)
                if warehouse[i, j] == 0:
                    color = BLACK
                else:
                    color = WHITE
                
                # Workstation color
                if (i, j) in warehouse_manager.workstations:
                    color = GREEN
                
                pygame.draw.rect(screen, color, cell_rect)
                
                # Draw order indicator
                if warehouse[i, j] == 1 and warehouse_manager.orders[i, j] == 1:
                    pygame.draw.rect(screen, RED, cell_rect)
                
                # Draw cell border
                pygame.draw.rect(screen, GRAY, cell_rect, 1)
        
        # Draw robots
        for robot in warehouse_manager.robots:
            draw_robot(screen, robot.x, robot.y, cell_size, margin,
                      robot.rotation, robot.status == RobotStatus.WORKING)
        
        # Draw UI elements
        pygame.draw.rect(screen, WHITE, order_button_rect)
        pygame.draw.rect(screen, WHITE, play_button_rect)
        pygame.draw.rect(screen, WHITE, evaluate_button_rect)
        pygame.draw.rect(screen, WHITE, slider_rect)
        
        # Draw button borders
        pygame.draw.rect(screen, BLACK, order_button_rect, 2)
        pygame.draw.rect(screen, BLACK, play_button_rect, 2)
        pygame.draw.rect(screen, BLACK, evaluate_button_rect, 2)
        pygame.draw.rect(screen, BLACK, slider_rect, 2)
        
        # Draw slider position
        slider_pos = int(slider_rect.x + (robot_count / max_robots) * slider_width)
        pygame.draw.circle(screen, BLACK, (slider_pos, slider_rect.centery), 8)
        
        # Render button texts
        order_text = font.render("Order Generation", True, BLACK)
        play_text = font.render("Play/Pause", True, BLACK)
        evaluate_text = font.render(f"Jobs: {warehouse_manager.completed_jobs}", True, BLACK)
        robot_text = font.render(f"Robots: {robot_count}", True, BLACK)
        time_text = font.render(f"Time: {warehouse_manager.time_step}s", True, BLACK)
        
        # Draw texts
        screen.blit(order_text, order_text.get_rect(center=order_button_rect.center))
        screen.blit(play_text, play_text.get_rect(center=play_button_rect.center))
        screen.blit(evaluate_text, evaluate_text.get_rect(center=evaluate_button_rect.center))
        screen.blit(robot_text, (slider_rect.x, slider_rect.y - 25))
        screen.blit(time_text, (margin, margin))
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()

if __name__ == '__main__':
    main()