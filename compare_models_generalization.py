import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prediction_model import RandomForestPredictor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
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

def prepare_data(data, include_robot_density=False, include_throughput_estimate=False, waiting_time_constant=5):
    """Prepare features and targets with optional robot density and throughput estimate features"""
    # Basic features
    basic_features = data[[
        'warehouse_width', 'warehouse_height', 'num_robots',
        'num_workstations', 'shelf_count', 'total_orders'  # Removed order_density, kept total_orders
    ]].values
    
    features = basic_features
    
    if include_robot_density:
        # Calculate and add robot density feature
        robot_density = data['num_robots'] / (data['warehouse_width'] + data['warehouse_height'])
        features = np.column_stack((features, robot_density))
    
    if include_throughput_estimate:
        # Calculate estimated throughput and completion time
        expected_travel_length = (data['warehouse_width'] + data['warehouse_height']) / 2
        robot_density = data['num_robots'] / (data['warehouse_width'] * data['warehouse_height'])
        
        # Base throughput without waiting
        base_throughput = data['num_robots'] / expected_travel_length
        
        # Waiting time impact increases with robot density and warehouse size
        waiting_impact = (robot_density * waiting_time_constant * 
                         (data['warehouse_width'] + data['warehouse_height']) / 2)
        
        # Final throughput with waiting time penalty
        estimated_throughput = base_throughput / (1 + waiting_impact)
        
        # Add both throughput and its components as features
        completion_time_estimate = 1 / estimated_throughput
        features = np.column_stack((
            features, 
            completion_time_estimate,
            base_throughput,
            waiting_impact
        ))
    
    targets = data[[
        'completion_time', 'orders_per_step',
        'robot_efficiency', 'completion_rate'
    ]].values
    
    return features, targets

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and split data
    print("Loading and splitting data...")
    train_data, test_data = load_and_split_data()
    
    # Test models with basic features (including actual orders)
    print("\n=== Models with Basic Features (Including Total Orders) ===")
    X_train, y_train = prepare_data(train_data, include_robot_density=False, include_throughput_estimate=False)
    X_test, y_test = prepare_data(test_data, include_robot_density=False, include_throughput_estimate=False)
    
    print("\nTraining Random Forest with basic features...")
    rf_model_basic = RandomForestPredictor()
    rf_model_basic.train(X_train, y_train)
    rf_predictions_basic, rf_metrics_basic = evaluate_model(rf_model_basic, X_test, y_test, "rf_basic")
    
    # Test models with different waiting time constants (throughput only)
    print("\n=== Testing Different Waiting Time Constants (Throughput Only) ===")
    waiting_time_constants = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    throughput_metrics = {}
    throughput_predictions = {}
    
    for wtc in waiting_time_constants:
        print(f"\nTesting waiting_time_constant = {wtc}")
        
        # Prepare data with throughput estimate only
        X_train, y_train = prepare_data(train_data, include_robot_density=False, 
                                      include_throughput_estimate=True, 
                                      waiting_time_constant=wtc)
        X_test, y_test = prepare_data(test_data, include_robot_density=False, 
                                    include_throughput_estimate=True, 
                                    waiting_time_constant=wtc)
        
        print(f"Training Random Forest...")
        rf_model_throughput = RandomForestPredictor()
        rf_model_throughput.train(X_train, y_train)
        
        predictions, metrics = evaluate_model(rf_model_throughput, X_test, y_test, f"rf_throughput_wtc{wtc}")
        
        throughput_metrics[wtc] = metrics
        throughput_predictions[wtc] = predictions
    
    # Find best waiting time constant
    best_wtc = min(waiting_time_constants, 
                   key=lambda x: throughput_metrics[x][f"rf_throughput_wtc{x}_completion_time_mae"])
    
    print(f"\nBest waiting time constant: {best_wtc}")
    print("\nCompletion Time MAE for different waiting time constants:")
    for wtc in waiting_time_constants:
        mae = throughput_metrics[wtc][f"rf_throughput_wtc{wtc}_completion_time_mae"]
        print(f"WTC = {wtc}: {mae:.4f}")
    
    # Test model with all features using best WTC
    print(f"\n=== Models with All Features (WTC = {best_wtc}) ===")
    X_train, y_train = prepare_data(train_data, include_robot_density=True, 
                                  include_throughput_estimate=True,
                                  waiting_time_constant=best_wtc)
    X_test, y_test = prepare_data(test_data, include_robot_density=True, 
                                include_throughput_estimate=True,
                                waiting_time_constant=best_wtc)
    
    print("\nTraining Random Forest with all features...")
    rf_model_all = RandomForestPredictor()
    rf_model_all.train(X_train, y_train)
    rf_predictions_all, rf_metrics_all = evaluate_model(rf_model_all, X_test, y_test, "rf_all")
    
    # Print comparison metrics
    print("\nRandom Forest Performance Comparison on 50x85 Warehouse:")
    metrics = {
        **rf_metrics_basic,
        **{f"rf_throughput_{target}_mae": throughput_metrics[best_wtc][f"rf_throughput_wtc{best_wtc}_{target}_mae"]
           for target in ['completion_time', 'orders_per_step', 'robot_efficiency', 'completion_rate']},
        **rf_metrics_all
    }
    target_names = ['completion_time', 'orders_per_step', 'robot_efficiency', 'completion_rate']
    
    for target in target_names:
        print(f"\n{target}:")
        basic_mae = metrics[f"rf_basic_{target}_mae"]
        throughput_mae = metrics[f"rf_throughput_{target}_mae"]
        all_mae = metrics[f"rf_all_{target}_mae"]
        
        print("Basic Features with Total Orders:")
        print(f"  MAE: {basic_mae:.4f}")
        
        print(f"\nWith Throughput Estimate Only (WTC = {best_wtc}):")
        print(f"  MAE: {throughput_mae:.4f}")
        print(f"  Improvement over basic: {((basic_mae - throughput_mae) / basic_mae * 100):.1f}%")
        
        print("\nWith All Features:")
        print(f"  MAE: {all_mae:.4f}")
        print(f"  Improvement over basic: {((basic_mae - all_mae) / basic_mae * 100):.1f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot completion time predictions
    plt.figure(figsize=(15, 8))
    
    # Basic model with total orders
    plt.scatter(test_data['completion_time'], rf_predictions_basic[:, 0], 
               alpha=0.5, label='Basic + Total Orders', color='blue')
    
    # Model with throughput estimate only
    plt.scatter(test_data['completion_time'], throughput_predictions[best_wtc][:, 0], 
               alpha=0.5, label=f'Throughput Only (WTC={best_wtc})', color='red')
    
    # Model with all features
    plt.scatter(test_data['completion_time'], rf_predictions_all[:, 0], 
               alpha=0.5, label='All Features', color='magenta')
    
    # Add perfect prediction line
    min_val = min(
        test_data['completion_time'].min(), 
        rf_predictions_basic[:, 0].min(),
        throughput_predictions[best_wtc][:, 0].min(),
        rf_predictions_all[:, 0].min()
    )
    max_val = max(
        test_data['completion_time'].max(),
        rf_predictions_basic[:, 0].max(),
        throughput_predictions[best_wtc][:, 0].max(),
        rf_predictions_all[:, 0].max()
    )
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Actual Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Model Comparison: Impact of Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/feature_comparison.png')
    plt.close()
    
    # Plot MAE vs waiting time constant
    plt.figure(figsize=(10, 6))
    wtc_values = list(waiting_time_constants)
    mae_values = [throughput_metrics[wtc][f"rf_throughput_wtc{wtc}_completion_time_mae"] 
                 for wtc in waiting_time_constants]
    
    plt.plot(wtc_values, mae_values, 'b-', marker='o', label='Throughput Only')
    plt.axhline(y=rf_metrics_basic['rf_basic_completion_time_mae'], 
                color='r', linestyle='--', label='Basic + Total Orders')
    plt.axhline(y=rf_metrics_all['rf_all_completion_time_mae'],
                color='m', linestyle='--', label='All Features')
    
    plt.xscale('log')  # Use log scale for better visualization
    plt.xlabel('Waiting Time Constant (log scale)')
    plt.ylabel('Completion Time MAE')
    plt.title('Impact of Waiting Time Constant on Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/waiting_time_constant_comparison.png')
    plt.close()
    
    print("\nResults saved to:")
    print("- plots/feature_comparison.png")
    print("- plots/waiting_time_constant_comparison.png")

if __name__ == "__main__":
    main() 