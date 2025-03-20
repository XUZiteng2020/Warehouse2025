import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from prediction_model import RandomForestPredictor
from generate_large_warehouse_data import LARGE_WAREHOUSE_DIR

# Constants
ORIGINAL_DATA_DIR = 'warehouse_data_files'

def load_trained_model():
    """Load and return the trained Random Forest model"""
    print("Loading trained Random Forest model...")
    
    # Load original training data to fit the scaler
    original_data = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, 'training_data.csv'))
    features = original_data[[
        'warehouse_width', 'warehouse_height', 'num_robots',
        'num_workstations', 'order_density', 'shelf_count'
    ]].values
    targets = original_data[[
        'completion_time', 'orders_per_step', 
        'robot_efficiency', 'completion_rate'
    ]].values
    
    # Create and train the model
    model = RandomForestPredictor()
    model.train(features, targets)
    
    return model

def evaluate_predictions(y_true, y_pred, warehouse_sizes):
    """Evaluate predictions and generate visualizations"""
    # Calculate MAE for each warehouse size
    size_mae = {}
    for size in warehouse_sizes:
        size_str = f"{size[0]}x{size[1]}"
        mask = (y_true['warehouse_width'] == size[0]) & (y_true['warehouse_height'] == size[1])
        mae = mean_absolute_error(y_true.loc[mask, 'completion_time'], 
                                y_pred[mask])
        size_mae[size_str] = mae
    
    # Overall MAE
    overall_mae = mean_absolute_error(y_true['completion_time'], y_pred)
    
    # Create scatter plot
    plt.figure(figsize=(12, 6))
    for size in warehouse_sizes:
        size_str = f"{size[0]}x{size[1]}"
        mask = (y_true['warehouse_width'] == size[0]) & (y_true['warehouse_height'] == size[1])
        plt.scatter(y_true.loc[mask, 'completion_time'], 
                   y_pred[mask],
                   label=size_str,
                   alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(y_true['completion_time'].min(), y_pred.min())
    max_val = max(y_true['completion_time'].max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Large Warehouse Completion Time: Ground Truth vs Predictions')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/large_warehouse_comparison.png')
    plt.close()
    
    # Create bar plot for MAE by warehouse size
    plt.figure(figsize=(10, 6))
    sizes = list(size_mae.keys())
    maes = list(size_mae.values())
    
    plt.bar(sizes, maes)
    plt.axhline(y=overall_mae, color='r', linestyle='--', 
                label=f'Overall MAE: {overall_mae:.2f}')
    
    plt.xlabel('Warehouse Size')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error by Warehouse Size')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('plots/large_warehouse_mae.png')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'warehouse_size': [f"{size[0]}x{size[1]}" for size in warehouse_sizes],
        'mae': [size_mae[f"{size[0]}x{size[1]}"] for size in warehouse_sizes],
        'num_samples': [sum((y_true['warehouse_width'] == size[0]) & 
                           (y_true['warehouse_height'] == size[1])) 
                       for size in warehouse_sizes]
    })
    results_df.to_csv('plots/large_warehouse_results.csv', index=False)
    
    return size_mae, overall_mae

def main():
    # Load the large warehouse data
    print("Loading large warehouse data...")
    large_data = pd.read_csv(os.path.join(LARGE_WAREHOUSE_DIR, 'training_data.csv'))
    
    # Load and use the trained model
    model = load_trained_model()
    
    # Make predictions
    print("Making predictions...")
    features = large_data[[
        'warehouse_width', 'warehouse_height', 'num_robots',
        'num_workstations', 'order_density', 'shelf_count'
    ]].values
    predictions = model.predict(features)
    
    # Save predictions
    large_data['predicted_completion_time'] = predictions[:, 0]
    large_data['predicted_orders_per_step'] = predictions[:, 1]
    large_data['predicted_robot_efficiency'] = predictions[:, 2]
    large_data['predicted_completion_rate'] = predictions[:, 3]
    large_data.to_csv(os.path.join(LARGE_WAREHOUSE_DIR, 'predictions.csv'), index=False)
    
    # Evaluate results
    print("\nEvaluating predictions...")
    warehouse_sizes = [(60, 100), (90, 150), (120, 200)]
    size_mae, overall_mae = evaluate_predictions(
        large_data,
        predictions[:, 0],  # Only completion time predictions
        warehouse_sizes
    )
    
    # Print results
    print("\nPrediction Error by Warehouse Size:")
    for size, mae in size_mae.items():
        print(f"{size}: MAE = {mae:.2f}")
    print(f"\nOverall MAE: {overall_mae:.2f}")
    
    print("\nFiles saved:")
    print(f"- {LARGE_WAREHOUSE_DIR}/predictions.csv (predictions)")
    print("- plots/large_warehouse_comparison.png (scatter plot)")
    print("- plots/large_warehouse_mae.png (error by warehouse size)")
    print("- plots/large_warehouse_results.csv (detailed results)")

if __name__ == "__main__":
    main() 