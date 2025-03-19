import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tqdm
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from functools import partial
import cudf
from cuml.ensemble import RandomForestRegressor as cuRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def generate_warehouse_layout(rows: int, cols: int) -> np.ndarray:
    """Generate a warehouse layout with the specified dimensions"""
    warehouse = np.ones((rows, cols))  # All shelves initially
    
    # Create aisles
    for i in range(1, rows-1, 3):
        warehouse[i, :] = 0  # Horizontal aisle
    for j in range(1, cols-1, 3):
        warehouse[:, j] = 0  # Vertical aisle
        
    return warehouse

def run_parallel_simulations(params: Tuple) -> Tuple[int, int, Dict]:
    """Run a single simulation with given parameters. Designed for parallel processing."""
    warehouse, robot_count, order_probability, max_steps = params
    
    # Simulate warehouse operations (simplified for demo)
    available_space = np.sum(warehouse == 0)
    total_orders = int(np.sum(warehouse == 1) * order_probability)
    
    # Estimate completion time based on robot count and orders
    completion_time = int((total_orders / robot_count) * (available_space ** 0.5))
    completion_time = min(completion_time, max_steps)
    
    completed_jobs = total_orders if completion_time < max_steps else int(total_orders * 0.8)
    
    metrics = {
        'robot_count': robot_count,
        'order_probability': order_probability,
        'steps': completion_time,
        'completed_jobs': completed_jobs,
        'is_timeout': completion_time >= max_steps
    }
    
    return completion_time, completed_jobs, metrics

def generate_training_data(
    num_samples: int = 1000,  # Increased sample size for better training
    warehouse_sizes: List[Tuple[int, int]] = [(10, 10), (15, 15), (20, 20), (25, 25)],
    robot_ratios: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # More granular ratios
    order_densities: List[float] = [0.2, 0.3, 0.4, 0.5],  # More density options
    max_steps: int = 5000,
    n_processes: Optional[int] = None
) -> pd.DataFrame:
    """Generate training data using parallel processing"""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"Generating training data using {n_processes} processes...")
    
    all_results = []
    
    # Create a pool of worker processes
    with mp.Pool(processes=n_processes) as pool:
        for size in warehouse_sizes:
            rows, cols = size
            print(f"\nProcessing warehouse size {size}")
            
            warehouse = generate_warehouse_layout(rows, cols)
            available_space = np.sum(warehouse == 0)
            
            # Calculate robot counts
            robot_counts = [max(1, int(available_space * ratio)) for ratio in robot_ratios]
            
            # Generate parameter combinations
            param_combinations = []
            for robot_count in robot_counts:
                for order_density in order_densities:
                    for _ in range(num_samples):
                        param_combinations.append((
                            warehouse.copy(),
                            robot_count,
                            order_density,
                            max_steps
                        ))
            
            # Run simulations in parallel
            print(f"Running {len(param_combinations)} simulations...")
            results = list(tqdm.tqdm(
                pool.imap(run_parallel_simulations, param_combinations),
                total=len(param_combinations)
            ))
            
            # Process results
            for _, _, metrics in results:
                metrics['warehouse_size'] = rows * cols
                metrics['warehouse_rows'] = rows
                metrics['warehouse_cols'] = cols
                all_results.append(metrics)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate additional features
    df['robot_density'] = df['robot_count'] / df['warehouse_size']
    df['completion_rate'] = df['completed_jobs'] / (df['warehouse_size'] * df['order_probability'])
    df['steps_per_job'] = df['steps'] / df['completed_jobs']
    
    print("\nTraining data generation complete!")
    print(f"Generated {len(df)} samples")
    print("\nSample statistics:")
    print(df.describe())
    
    return df

def train_gpu_model(data: pd.DataFrame) -> Tuple[cuRFRegressor, Dict]:
    """Train a GPU-accelerated RandomForest model to predict completion time"""
    features = [
        'warehouse_size', 'warehouse_rows', 'warehouse_cols',
        'robot_count', 'robot_density', 'order_probability'
    ]
    
    X = data[features]
    y = data['steps']
    
    # Convert to cuDF for GPU processing
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_cudf, y_cudf, test_size=0.2, random_state=42)
    
    # Train model on GPU
    print("\nTraining GPU-accelerated RandomForest model...")
    model = cuRFRegressor(
        n_estimators=100,
        max_depth=None,
        n_streams=4,  # Utilize multiple GPU streams
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test.values, y_pred.values)
    rmse = np.sqrt(mse)
    
    print("\nModel Performance:")
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    print(f"Root Mean Squared Error: {rmse:.2f} steps")
    
    # Get feature importance
    importance = dict(zip(features, model.feature_importances_))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance in Completion Time Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    
    return model, importance

def validate_gpu_model(
    model: cuRFRegressor,
    num_validation_samples: int = 100,  # Increased validation samples
    warehouse_sizes: List[Tuple[int, int]] = [(30, 30), (35, 35)],  # Test on larger warehouses
    max_steps: int = 5000,
    n_processes: Optional[int] = None
) -> None:
    """Validate model predictions using GPU acceleration"""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"\nValidating model predictions using {n_processes} processes...")
    
    validation_results = []
    
    with mp.Pool(processes=n_processes) as pool:
        for size in warehouse_sizes:
            rows, cols = size
            print(f"\nValidating warehouse size {size}")
            
            warehouse = generate_warehouse_layout(rows, cols)
            available_space = np.sum(warehouse == 0)
            
            # Test different configurations
            robot_counts = [int(available_space * ratio) for ratio in [0.3, 0.5, 0.7]]
            order_densities = [0.3, 0.5]
            
            param_combinations = []
            for robot_count in robot_counts:
                for order_density in order_densities:
                    for _ in range(num_validation_samples):
                        param_combinations.append((
                            warehouse.copy(),
                            robot_count,
                            order_density,
                            max_steps
                        ))
            
            print(f"Running {len(param_combinations)} validation simulations...")
            results = list(tqdm.tqdm(
                pool.imap(run_parallel_simulations, param_combinations),
                total=len(param_combinations)
            ))
            
            for _, _, metrics in results:
                features = cudf.DataFrame([{
                    'warehouse_size': rows * cols,
                    'warehouse_rows': rows,
                    'warehouse_cols': cols,
                    'robot_count': metrics['robot_count'],
                    'robot_density': metrics['robot_count'] / (rows * cols),
                    'order_probability': metrics['order_probability']
                }])
                
                predicted_steps = model.predict(features)[0]
                
                validation_results.append({
                    'actual_steps': metrics['steps'],
                    'predicted_steps': predicted_steps,
                    'error': predicted_steps - metrics['steps'],
                    'robot_count': metrics['robot_count'],
                    'order_probability': metrics['order_probability'],
                    'warehouse_size': rows * cols
                })
    
    validation_df = pd.DataFrame(validation_results)
    
    mse = mean_squared_error(validation_df['actual_steps'], validation_df['predicted_steps'])
    rmse = np.sqrt(mse)
    r2 = r2_score(validation_df['actual_steps'], validation_df['predicted_steps'])
    
    print("\nValidation Results:")
    print(f"Root Mean Squared Error: {rmse:.2f} steps")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(validation_df['actual_steps'], validation_df['predicted_steps'], alpha=0.5)
    plt.plot([0, max(validation_df['actual_steps'])], [0, max(validation_df['actual_steps'])], 'r--')
    plt.xlabel('Actual Steps')
    plt.ylabel('Predicted Steps')
    plt.title('Model Predictions vs Actual Completion Times')
    plt.tight_layout()
    plt.savefig('prediction_validation.png', dpi=300, bbox_inches='tight')
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(validation_df['error'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Prediction Error (steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_errors.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Generate training data
    print("\nGenerating training data...")
    training_data = generate_training_data(
        num_samples=1000,  # Increased for better results
        warehouse_sizes=[(10, 10), (15, 15), (20, 20), (25, 25)],
        robot_ratios=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        order_densities=[0.2, 0.3, 0.4, 0.5]
    )
    
    # Train GPU-accelerated model
    print("\nTraining model...")
    model, feature_importance = train_gpu_model(training_data)
    
    # Validate model
    print("\nValidating model...")
    validate_gpu_model(
        model,
        num_validation_samples=100,
        warehouse_sizes=[(30, 30), (35, 35)]
    ) 