import pygame
import numpy as np
import os
import random
from order_generation import uniform_order_distribution
from warehouse_manager import WarehouseManager
from robot import (
    RobotStatus, 
    build_reservation_table,
    update_reservation_table,
    time_expanded_a_star,
    is_cell_safe
)
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

def count_shelves(warehouse):
    """Count the number of shelf cells (value 1) in the warehouse"""
    return np.sum(warehouse == 1)

class WarehouseVisualizer:
    def __init__(self, warehouse, collision_method=1):
        pygame.init()
        self.warehouse = warehouse
        self.rows, self.cols = warehouse.shape
        self.collision_method = collision_method  # Store current collision method
        
        # Screen settings
        self.screen_width = 1600  # Fixed window size
        self.screen_height = 900
        self.margin = 20
        
        # UI element dimensions
        self.button_width = 200
        self.button_height = 50
        self.slider_width = 200
        self.slider_height = 20
        
        # Calculate initial cell size to fit the whole warehouse
        self.base_cell_size = min(
            (self.screen_width - self.button_width - 3 * self.margin) / self.cols,
            (self.screen_height - 2 * self.margin) / self.rows
        )
        
        # Zoom and pan settings
        self.zoom_level = 1.0
        self.min_zoom = self.base_cell_size / 40
        self.max_zoom = 2.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.last_mouse_pos = None
        
        # Initialize screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Warehouse Simulation")
        
        # Initialize warehouse manager with specified collision method
        self.warehouse_manager = WarehouseManager(warehouse, collision_method=collision_method)
        self.robot_count = max(1, int(np.sqrt(np.sum(warehouse == 1))))
        self.warehouse_manager.initialize_robots(self.robot_count)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.GRAY = (200, 200, 200)
        self.BLUE = (0, 0, 255)
        
        # Font
        self.font = pygame.font.SysFont(None, 30)
        
        # Initialize UI elements
        self.init_ui_elements()
    
    def init_ui_elements(self):
        """Initialize UI element rectangles"""
        # Add collision method toggle button
        self.collision_button_rect = pygame.Rect(
            self.screen_width - self.button_width - self.margin,
            5 * self.margin + 4 * self.button_height,
            self.button_width,
            self.button_height
        )
        
        # Existing UI elements
        self.order_button_rect = pygame.Rect(
            self.screen_width - self.button_width - self.margin, self.margin,
            self.button_width, self.button_height
        )
        self.play_button_rect = pygame.Rect(
            self.screen_width - self.button_width - self.margin, 2 * self.margin + self.button_height,
            self.button_width, self.button_height
        )
        self.evaluate_button_rect = pygame.Rect(
            self.screen_width - self.button_width - self.margin, 3 * self.margin + 2 * self.button_height,
            self.button_width, self.button_height
        )
        self.slider_rect = pygame.Rect(
            self.screen_width - self.button_width - self.margin, 4 * self.margin + 3 * self.button_height,
            self.slider_width, self.slider_height
        )
    
    def get_cell_size(self):
        """Get current cell size based on zoom level"""
        return self.base_cell_size * self.zoom_level
    
    def screen_to_grid(self, screen_x, screen_y):
        """Convert screen coordinates to grid coordinates"""
        cell_size = self.get_cell_size()
        grid_x = int((screen_y - self.margin - self.pan_y) / cell_size)
        grid_y = int((screen_x - self.margin - self.pan_x) / cell_size)
        return grid_x, grid_y
    
    def grid_to_screen(self, grid_x, grid_y):
        """Convert grid coordinates to screen coordinates"""
        cell_size = self.get_cell_size()
        screen_x = grid_y * cell_size + self.margin + self.pan_x
        screen_y = grid_x * cell_size + self.margin + self.pan_y
        return screen_x, screen_y
    
    def draw_robot(self, x, y, rotation, is_working):
        """Draw a robot with its orientation arrow and working status indicator"""
        cell_size = self.get_cell_size()
        screen_x, screen_y = self.grid_to_screen(x, y)
        
        # Check if robot is visible on screen
        if (screen_x + cell_size < 0 or screen_x > self.screen_width or
            screen_y + cell_size < 0 or screen_y > self.screen_height):
            return
        
        robot_size = int(cell_size * 0.7)
        robot_margin = (cell_size - robot_size) // 2
        
        # Robot body
        robot_rect = pygame.Rect(
            screen_x + robot_margin,
            screen_y + robot_margin,
            robot_size, robot_size
        )
        pygame.draw.rect(self.screen, (255, 165, 0), robot_rect)
        
        # Direction arrow
        center_x = screen_x + cell_size // 2
        center_y = screen_y + cell_size // 2
        arrow_length = robot_size // 2
        
        end_x = center_x + arrow_length * math.cos(math.radians(rotation))
        end_y = center_y - arrow_length * math.sin(math.radians(rotation))
        
        pygame.draw.line(self.screen, self.BLACK, (center_x, center_y), (end_x, end_y), 2)
        
        # Working status indicator
        if is_working:
            dot_radius = robot_size // 6
            pygame.draw.circle(self.screen, self.RED,
                             (center_x, screen_y + robot_margin // 2),
                             dot_radius)
    
    def draw_warehouse(self):
        """Draw the warehouse grid with current zoom and pan"""
        cell_size = self.get_cell_size()
        
        # Calculate visible range of cells
        start_x = max(0, int((-self.pan_y - self.margin) / cell_size))
        end_x = min(self.rows, int((-self.pan_y + self.screen_height - self.margin) / cell_size) + 1)
        start_y = max(0, int((-self.pan_x - self.margin) / cell_size))
        end_y = min(self.cols, int((-self.pan_x + self.screen_width - self.margin) / cell_size) + 1)
        
        # Draw visible cells
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                screen_x, screen_y = self.grid_to_screen(i, j)
                cell_rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size)
                
                # Base color
                if self.warehouse[i, j] == 0:
                    color = self.BLACK
                else:
                    color = self.WHITE
                
                # Workstation color
                if (i, j) in self.warehouse_manager.workstations:
                    color = self.GREEN
                
                pygame.draw.rect(self.screen, color, cell_rect)
                
                # Draw order indicator
                if (self.warehouse[i, j] == 1 and 
                    self.warehouse_manager.orders[i, j] == 1):
                    pygame.draw.rect(self.screen, self.RED, cell_rect)
                
                # Draw cell border
                pygame.draw.rect(self.screen, self.GRAY, cell_rect, 1)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Draw buttons
        for button_rect in [self.order_button_rect, self.play_button_rect, 
                          self.evaluate_button_rect, self.collision_button_rect]:
            pygame.draw.rect(self.screen, self.WHITE, button_rect)
            pygame.draw.rect(self.screen, self.BLACK, button_rect, 2)
        
        # Draw slider
        pygame.draw.rect(self.screen, self.WHITE, self.slider_rect)
        pygame.draw.rect(self.screen, self.BLACK, self.slider_rect, 2)
        
        # Draw slider position
        max_robots = max(1, int(np.sqrt(np.sum(self.warehouse == 1))))
        slider_pos = int(self.slider_rect.x + 
                        (self.robot_count / max_robots) * self.slider_width)
        pygame.draw.circle(self.screen, self.BLACK, 
                         (slider_pos, self.slider_rect.centery), 8)
        
        # Draw texts
        order_text = self.font.render("Order Generation", True, self.BLACK)
        play_text = self.font.render("Play/Pause", True, self.BLACK)
        evaluate_text = self.font.render(
            f"Jobs: {self.warehouse_manager.completed_jobs}", True, self.BLACK
        )
        robot_text = self.font.render(f"Robots: {self.robot_count}", True, self.BLACK)
        time_text = self.font.render(
            f"Time: {self.warehouse_manager.time_step}s", True, self.BLACK
        )
        zoom_text = self.font.render(f"Zoom: {self.zoom_level:.1f}x", True, self.BLACK)
        collision_text = self.font.render(
            "Collision: Wait" if self.collision_method == 1 else "Collision: Reserve",
            True, self.BLACK
        )
        
        # Position texts
        self.screen.blit(order_text, order_text.get_rect(
            center=self.order_button_rect.center))
        self.screen.blit(play_text, play_text.get_rect(
            center=self.play_button_rect.center))
        self.screen.blit(evaluate_text, evaluate_text.get_rect(
            center=self.evaluate_button_rect.center))
        self.screen.blit(robot_text, (self.slider_rect.x, self.slider_rect.y - 25))
        self.screen.blit(time_text, (self.margin, self.margin))
        self.screen.blit(zoom_text, (self.margin, self.margin + 30))
        self.screen.blit(collision_text, collision_text.get_rect(
            center=self.collision_button_rect.center))
    
    def toggle_collision_method(self):
        """Toggle between collision methods and reinitialize robots"""
        self.collision_method = 2 if self.collision_method == 1 else 1
        self.warehouse_manager = WarehouseManager(self.warehouse, collision_method=self.collision_method)
        self.warehouse_manager.initialize_robots(self.robot_count)
        print(f"Switched to collision method {self.collision_method}")
    
    def handle_input(self, event):
        """Handle input events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (4, 5):  # Mouse wheel
                # Calculate zoom center (mouse position)
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x < self.screen_width - self.button_width - 2 * self.margin:
                    old_zoom = self.zoom_level
                    if event.button == 4:  # Zoom in
                        self.zoom_level = min(self.zoom_level * 1.1, self.max_zoom)
                    else:  # Zoom out
                        self.zoom_level = max(self.zoom_level / 1.1, self.min_zoom)
                    
                    # Adjust pan to keep mouse position fixed
                    zoom_factor = self.zoom_level / old_zoom
                    self.pan_x = mouse_x - (mouse_x - self.pan_x) * zoom_factor
                    self.pan_y = mouse_y - (mouse_y - self.pan_y) * zoom_factor
            
            elif event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                
                # Handle UI element clicks
                if self.order_button_rect.collidepoint(mouse_pos):
                    orders = uniform_order_distribution(0.3, self.warehouse)
                    self.warehouse_manager.update_orders(orders)
                elif self.play_button_rect.collidepoint(mouse_pos):
                    self.warehouse_manager.toggle_play()
                elif self.collision_button_rect.collidepoint(mouse_pos):
                    self.toggle_collision_method()
                elif self.slider_rect.collidepoint(mouse_pos):
                    self.dragging = True
                else:
                    # Start panning
                    self.dragging = True
                    self.last_mouse_pos = mouse_pos
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.last_mouse_pos = None
        
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_pos = pygame.mouse.get_pos()
            if self.slider_rect.collidepoint(mouse_pos):
                # Update robot count
                max_robots = max(1, int(np.sqrt(np.sum(self.warehouse == 1))))
                relative_x = min(max(mouse_pos[0] - self.slider_rect.x, 0), 
                               self.slider_width)
                new_robot_count = max(1, int((relative_x / self.slider_width) * max_robots))
                if new_robot_count != self.robot_count:
                    self.robot_count = new_robot_count
                    self.warehouse_manager.initialize_robots(self.robot_count)
            elif self.last_mouse_pos:
                # Pan the view
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                self.pan_x += dx
                self.pan_y += dy
                self.last_mouse_pos = mouse_pos
    
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        frame_count = 0
        
        while running:
            frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_input(event)
            
            # Update warehouse state
            if self.warehouse_manager.update() and frame_count % 60 == 0:
                print(f"Time step: {self.warehouse_manager.time_step}")
            
            # Draw everything
            self.screen.fill(self.GRAY)
            self.draw_warehouse()
            
            # Draw robots
            for robot in self.warehouse_manager.robots:
                self.draw_robot(robot.x, robot.y, robot.rotation,
                              robot.status == RobotStatus.WORKING)
            
            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def main():
    # Load warehouse data
    warehouse_dir = 'warehouse_data_files'
    warehouse_files = [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')]
    random_warehouse_file = os.path.join(warehouse_dir, random.choice(warehouse_files))
    warehouse = np.loadtxt(random_warehouse_file)
    
    # Create and run visualizer with specified collision method
    # Use collision_method=1 for 10-second waiting rule
    # Use collision_method=2 for reservation-based system
    visualizer = WarehouseVisualizer(warehouse, collision_method=1)  # Change this value to switch methods
    visualizer.run()

if __name__ == "__main__":
    main()