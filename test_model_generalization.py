import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from prediction_model import RandomForestPredictor
import os

def load_and_split_data():
    """Load data and split into training (small) and test (medium) warehouses"""
    data = pd.read_csv('warehouse_data_files/training_data.csv')
    
    # Split data by warehouse size
    train_mask = (
        ((data['warehouse_width'] == 30) & (data['warehouse_height'] == 50)) |
        ((data['warehouse_width'] == 40) & (data['warehouse_height'] == 70))
    )
    test_mask = (
        (data['warehouse_width'] == 50) & (data['warehouse_height'] == 85)
    )
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    print(f"Training data size: {len(train_data)} samples")
    print(f"Test data size: {len(test_data)} samples")
    
    return train_data, test_data

def train_model(train_data):
    """Train Random Forest model on small warehouse data"""
    # Prepare features and targets
    features = train_data[[
        'warehouse_width', 'warehouse_height', 'num_robots',
        'num_workstations', 'order_density', 'shelf_count'
    ]].values
    
    targets = train_data[[
        'completion_time', 'orders_per_step',
        'robot_efficiency', 'completion_rate'
    ]].values
    
    # Train model
    model = RandomForestPredictor()
    model.train(features, targets)
    
    return model

def evaluate_model(model, test_data):
    """Evaluate model performance on medium warehouse data"""
    # Prepare test features
    test_features = test_data[[
        'warehouse_width', 'warehouse_height', 'num_robots',
        'num_workstations', 'order_density', 'shelf_count'
    ]].values
    
    # Make predictions
    predictions = model.predict(test_features)
    
    # Calculate metrics for each target variable
    target_names = ['completion_time', 'orders_per_step', 'robot_efficiency', 'completion_rate']
    metrics = {}
    
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(test_data[target], predictions[:, i])
        metrics[target] = mae
    
    return predictions, metrics

def plot_results(test_data, predictions):
    """Create visualization of predictions vs actual values"""
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot completion time comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(test_data['completion_time'], predictions[:, 0], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(test_data['completion_time'].min(), predictions[:, 0].min())
    max_val = max(test_data['completion_time'].max(), predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Model Generalization: 50x85 Warehouse Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/generalization_test.png')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'actual_completion_time': test_data['completion_time'],
        'predicted_completion_time': predictions[:, 0],
        'actual_orders_per_step': test_data['orders_per_step'],
        'predicted_orders_per_step': predictions[:, 1],
        'actual_robot_efficiency': test_data['robot_efficiency'],
        'predicted_robot_efficiency': predictions[:, 2],
        'actual_completion_rate': test_data['completion_rate'],
        'predicted_completion_rate': predictions[:, 3],
        'num_robots': test_data['num_robots'],
        'num_workstations': test_data['num_workstations'],
        'order_density': test_data['order_density']
    })
    results_df.to_csv('plots/generalization_results.csv', index=False)

def main():
    # Load and split data
    print("Loading and splitting data...")
    train_data, test_data = load_and_split_data()
    
    # Train model on small warehouses
    print("\nTraining model on small warehouses (30x50, 40x70)...")
    model = train_model(train_data)
    
    # Evaluate on medium warehouse
    print("\nEvaluating model on medium warehouse (50x85)...")
    predictions, metrics = evaluate_model(model, test_data)
    
    # Print metrics
    print("\nModel Performance on 50x85 Warehouse:")
    for target, mae in metrics.items():
        print(f"{target}: MAE = {mae:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_results(test_data, predictions)
    
    print("\nResults saved to:")
    print("- plots/generalization_test.png")
    print("- plots/generalization_results.csv")

if __name__ == "__main__":
    main() 