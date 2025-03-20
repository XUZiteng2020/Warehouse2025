import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tqdm
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from warehouse_manager import WarehouseManager
from robot import Robot, RobotStatus
import os

# Set plotting styles
plt.style.use('default')  # Reset to default style
sns.set_style("whitegrid")  # Set seaborn style

def generate_warehouse_layout(width: int, height: int) -> np.ndarray:
    """
    Generate a warehouse layout with the following pattern:
    - 4 empty columns at the start
    - 5x2 shelf blocks
    - 2-cell wide aisles between shelf blocks
    - 2 empty rows at the top
    
    Args:
        width: Width of the warehouse
        height: Height of the warehouse
        
    Returns:
        np.ndarray: 2D array representing the warehouse layout (0: empty, 1: shelf)
    """
    # Initialize empty warehouse
    layout = np.zeros((height, width), dtype=int)
    
    # Start placing shelves after initial empty space
    start_row = 2  # 2 empty rows at top
    start_col = 4  # 4 empty columns at start
    
    # Place shelves in a grid pattern
    shelf_block_width = 5
    shelf_block_height = 2
    aisle_width = 2
    
    row = start_row
    while row + shelf_block_height <= height:
        col = start_col
        while col + shelf_block_width <= width:
            # Place a block of shelves
            layout[row:row + shelf_block_height, col:col + shelf_block_width] = 1
            col += shelf_block_width + aisle_width
        row += shelf_block_height + aisle_width
    
    return layout

def generate_ml_dataset(
    num_samples: int = 100,
    warehouse_sizes: List[Tuple[int, int]] = [(10, 20), (15, 30), (20, 40)],
    order_densities: List[float] = [0.1, 0.2, 0.3],
    robot_counts: List[int] = [2, 3, 4, 5],
    workstation_counts: List[int] = [2, 3, 4],
    max_steps: int = 5000,
    save_path: str = 'warehouse_data_files/training_data.csv'
) -> pd.DataFrame:
    """
    Generate a dataset for machine learning with specified features and target variable.
    
    Features (X):
    - warehouse_width: Width of the warehouse
    - warehouse_height: Height of the warehouse
    - order_density: Probability of order generation per step
    - actual_orders: Number of orders actually generated
    - num_robots: Number of robots in the warehouse
    - num_workstations: Number of workstations
    
    Target (Y):
    - completion_time: Steps taken to complete all orders
    
    Returns:
        pd.DataFrame: DataFrame containing features and target variable
    """
    data = []
    total_combinations = len(warehouse_sizes) * len(order_densities) * len(robot_counts) * len(workstation_counts) * num_samples
    
    with tqdm.tqdm(total=total_combinations, desc="Generating dataset") as pbar:
        for warehouse_size in warehouse_sizes:
            width, height = warehouse_size
            layout = generate_warehouse_layout(width, height)
            
            for order_density in order_densities:
                for num_robots in robot_counts:
                    for num_workstations in workstation_counts:
                        for _ in range(num_samples):
                            # Initialize warehouse manager
                            manager = WarehouseManager(
                                layout=layout.copy(),
                                num_robots=num_robots,
                                num_workstations=num_workstations,
                                order_probability=order_density
                            )
                            
                            # Run simulation
                            step = 0
                            while step < max_steps and not manager.all_orders_completed():
                                manager.update()
                                step += 1
                            
                            # Collect data
                            data.append({
                                'warehouse_width': width,
                                'warehouse_height': height,
                                'order_density': order_density,
                                'actual_orders': len(manager.completed_orders) + len(manager.pending_orders),
                                'num_robots': num_robots,
                                'num_workstations': num_workstations,
                                'completion_time': step if manager.all_orders_completed() else max_steps
                            })
                            pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: completion_time")
    print(f"\nFeature ranges:")
    for col in df.columns:
        print(f"{col}: [{df[col].min()}, {df[col].max()}]")
    
    return df

def train_and_evaluate_model(df: pd.DataFrame, 
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[RandomForestRegressor, Dict, pd.DataFrame]:
    """
    Train and evaluate a Random Forest model for predicting completion time.
    
    Args:
        df: DataFrame containing the features and target
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        model: Trained RandomForestRegressor
        metrics: Dictionary containing evaluation metrics
        feature_importance: DataFrame with feature importance scores
    """
    # Prepare features and target
    X = df.drop('completion_time', axis=1)
    y = df['completion_time']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    metrics['cv_r2_mean'] = cv_scores.mean()
    metrics['cv_r2_std'] = cv_scores.std()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, metrics, feature_importance

def plot_model_results(df: pd.DataFrame, 
                      model: RandomForestRegressor, 
                      metrics: Dict,
                      feature_importance: pd.DataFrame,
                      save_dir: str = 'warehouse_data_files'):
    """
    Create visualizations for model evaluation.
    """
    # Prepare predictions for plotting
    X = df.drop('completion_time', axis=1)
    y_true = df['completion_time']
    y_pred = model.predict(X)
    
    # Create figure with subplots
    plt.figure(figsize=(20, 12))
    
    # 1. Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Actual vs Predicted Completion Time')
    
    # 2. Feature Importance
    plt.subplot(2, 2, 2)
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    
    # 3. Prediction Error Distribution
    plt.subplot(2, 2, 3)
    error = y_pred - y_true
    sns.histplot(error, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    
    # 4. Error vs Predicted Value
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, abs(error), alpha=0.5)
    plt.xlabel('Predicted Value')
    plt.ylabel('Absolute Error')
    plt.title('Error Magnitude vs Predicted Value')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_evaluation.png')
    plt.close()

def main():
    """Main function to generate dataset and train model"""
    print("Generating machine learning dataset...")
    start_time = time.time()
    
    # Generate dataset
    df = generate_ml_dataset(
        num_samples=50,  # Reduced for faster execution
        warehouse_sizes=[(10, 20), (15, 30), (20, 40)],
        order_densities=[0.1, 0.2, 0.3],
        robot_counts=[2, 3, 4, 5],
        workstation_counts=[2, 3, 4],
        max_steps=5000
    )
    
    # Train and evaluate model
    model, metrics, feature_importance = train_and_evaluate_model(df)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Training R² Score: {metrics['train_r2']:.4f}")
    print(f"Testing R² Score: {metrics['test_r2']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Testing RMSE: {metrics['test_rmse']:.4f}")
    print(f"Training MAE: {metrics['train_mae']:.4f}")
    print(f"Testing MAE: {metrics['test_mae']:.4f}")
    print(f"\nCross-validation R² Score: {metrics['cv_r2_mean']:.4f} (±{metrics['cv_r2_std']:.4f})")
    
    # Print feature importance
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    # Create visualizations
    plot_model_results(df, model, metrics, feature_importance)
    
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    print(f"\nData saved to: warehouse_data_files/training_data.csv")
    print(f"Model evaluation plots saved to: warehouse_data_files/model_evaluation.png")

if __name__ == "__main__":
    main() 