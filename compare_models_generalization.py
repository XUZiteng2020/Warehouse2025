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

class NeuralNetworkPredictor:
    def __init__(self):
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def create_model(self, input_dim):
        """Create neural network architecture with regularization"""
        regularizer = l1_l2(l1=0.01, l2=0.01)
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim,
                  kernel_regularizer=regularizer),
            Dropout(0.3),  # Increased dropout
            Dense(64, activation='relu',
                  kernel_regularizer=regularizer),
            Dropout(0.3),
            Dense(32, activation='relu',
                  kernel_regularizer=regularizer),
            Dense(4)  # 4 outputs: completion_time, orders_per_step, robot_efficiency, completion_rate
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X, y):
        """Train the neural network"""
        # Scale features and targets
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create and train model
        self.model = self.create_model(X.shape[1])
        
        # Added early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X_scaled, y_scaled,
            epochs=200,  # Increased epochs since we have early stopping
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler_x.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        return self.scaler_y.inverse_transform(y_pred_scaled)

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

def prepare_data(data, include_robot_density=False):
    """Prepare features and targets with optional robot density feature"""
    # Basic features
    basic_features = data[[
        'warehouse_width', 'warehouse_height', 'num_robots',
        'num_workstations', 'order_density', 'shelf_count'
    ]].values
    
    if include_robot_density:
        # Calculate and add robot density feature
        robot_density = data['num_robots'] / (data['warehouse_width'] + data['warehouse_height'])
        features = np.column_stack((basic_features, robot_density))
    else:
        features = basic_features
    
    targets = data[[
        'completion_time', 'orders_per_step',
        'robot_efficiency', 'completion_rate'
    ]].values
    
    return features, targets

def evaluate_model(model, features, targets, model_name):
    """Evaluate model performance"""
    predictions = model.predict(features)
    
    # Calculate metrics for each target variable
    target_names = ['completion_time', 'orders_per_step', 'robot_efficiency', 'completion_rate']
    metrics = {}
    
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(targets[:, i], predictions[:, i])
        metrics[f"{model_name}_{target}_mae"] = mae
    
    return predictions, metrics

def plot_comparison(test_data, rf_predictions, nn_predictions):
    """Create visualization comparing both models"""
    os.makedirs('plots', exist_ok=True)
    
    # Plot completion time predictions
    plt.figure(figsize=(12, 6))
    
    # Random Forest predictions
    plt.scatter(test_data['completion_time'], rf_predictions[:, 0], 
               alpha=0.5, label='Random Forest', color='blue')
    
    # Neural Network predictions
    plt.scatter(test_data['completion_time'], nn_predictions[:, 0], 
               alpha=0.5, label='Neural Network', color='red')
    
    # Add perfect prediction line
    min_val = min(test_data['completion_time'].min(), 
                 rf_predictions[:, 0].min(),
                 nn_predictions[:, 0].min())
    max_val = max(test_data['completion_time'].max(),
                 rf_predictions[:, 0].max(),
                 nn_predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Actual Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Model Comparison: 50x85 Warehouse Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'actual_completion_time': test_data['completion_time'],
        'rf_predicted_completion_time': rf_predictions[:, 0],
        'nn_predicted_completion_time': nn_predictions[:, 0],
        'actual_orders_per_step': test_data['orders_per_step'],
        'rf_predicted_orders_per_step': rf_predictions[:, 1],
        'nn_predicted_orders_per_step': nn_predictions[:, 1],
        'actual_robot_efficiency': test_data['robot_efficiency'],
        'rf_predicted_robot_efficiency': rf_predictions[:, 2],
        'nn_predicted_robot_efficiency': nn_predictions[:, 2],
        'actual_completion_rate': test_data['completion_rate'],
        'rf_predicted_completion_rate': rf_predictions[:, 3],
        'nn_predicted_completion_rate': nn_predictions[:, 3],
        'num_robots': test_data['num_robots'],
        'num_workstations': test_data['num_workstations'],
        'order_density': test_data['order_density'],
        'robot_density': test_data['num_robots'] / (test_data['warehouse_width'] + test_data['warehouse_height'])
    })
    results_df.to_csv('plots/model_comparison_results.csv', index=False)

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and split data
    print("Loading and splitting data...")
    train_data, test_data = load_and_split_data()
    
    # Test models without robot density feature
    print("\n=== Models without Robot Density Feature ===")
    
    # Prepare data without robot density
    X_train, y_train = prepare_data(train_data, include_robot_density=False)
    X_test, y_test = prepare_data(test_data, include_robot_density=False)
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest on small warehouses...")
    rf_model_basic = RandomForestPredictor()
    rf_model_basic.train(X_train, y_train)
    
    print("Evaluating Random Forest on medium warehouse...")
    rf_predictions_basic, rf_metrics_basic = evaluate_model(rf_model_basic, X_test, y_test, "rf_basic")
    
    # Train and evaluate Neural Network
    print("\nTraining Neural Network on small warehouses...")
    nn_model_basic = NeuralNetworkPredictor()
    nn_model_basic.train(X_train, y_train)
    
    print("Evaluating Neural Network on medium warehouse...")
    nn_predictions_basic, nn_metrics_basic = evaluate_model(nn_model_basic, X_test, y_test, "nn_basic")
    
    # Test models with robot density feature
    print("\n=== Models with Robot Density Feature ===")
    
    # Prepare data with robot density
    X_train, y_train = prepare_data(train_data, include_robot_density=True)
    X_test, y_test = prepare_data(test_data, include_robot_density=True)
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest on small warehouses...")
    rf_model = RandomForestPredictor()
    rf_model.train(X_train, y_train)
    
    print("Evaluating Random Forest on medium warehouse...")
    rf_predictions, rf_metrics = evaluate_model(rf_model, X_test, y_test, "rf")
    
    # Train and evaluate Neural Network
    print("\nTraining Neural Network on small warehouses...")
    nn_model = NeuralNetworkPredictor()
    nn_model.train(X_train, y_train)
    
    print("Evaluating Neural Network on medium warehouse...")
    nn_predictions, nn_metrics = evaluate_model(nn_model, X_test, y_test, "nn")
    
    # Print comparison metrics
    print("\nModel Performance Comparison on 50x85 Warehouse:")
    metrics = {**rf_metrics_basic, **nn_metrics_basic, **rf_metrics, **nn_metrics}
    target_names = ['completion_time', 'orders_per_step', 'robot_efficiency', 'completion_rate']
    
    for target in target_names:
        print(f"\n{target}:")
        rf_basic_mae = metrics[f"rf_basic_{target}_mae"]
        nn_basic_mae = metrics[f"nn_basic_{target}_mae"]
        rf_mae = metrics[f"rf_{target}_mae"]
        nn_mae = metrics[f"nn_{target}_mae"]
        
        print("Without Robot Density Feature:")
        print(f"  Random Forest MAE: {rf_basic_mae:.4f}")
        print(f"  Neural Network MAE: {nn_basic_mae:.4f}")
        print(f"  Difference: {((rf_basic_mae - nn_basic_mae) / rf_basic_mae * 100):.1f}%")
        
        print("\nWith Robot Density Feature:")
        print(f"  Random Forest MAE: {rf_mae:.4f}")
        print(f"  Neural Network MAE: {nn_mae:.4f}")
        print(f"  Difference: {((rf_mae - nn_mae) / rf_mae * 100):.1f}%")
        
        print(f"\nFeature Impact:")
        print(f"  Random Forest Improvement: {((rf_basic_mae - rf_mae) / rf_basic_mae * 100):.1f}%")
        print(f"  Neural Network Improvement: {((nn_basic_mae - nn_mae) / nn_basic_mae * 100):.1f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot completion time predictions for all models
    plt.figure(figsize=(15, 8))
    
    # Basic models
    plt.scatter(test_data['completion_time'], rf_predictions_basic[:, 0], 
               alpha=0.5, label='Random Forest (Basic)', color='blue')
    plt.scatter(test_data['completion_time'], nn_predictions_basic[:, 0], 
               alpha=0.5, label='Neural Network (Basic)', color='red')
    
    # Models with robot density
    plt.scatter(test_data['completion_time'], rf_predictions[:, 0], 
               alpha=0.5, label='Random Forest (with Robot Density)', color='cyan')
    plt.scatter(test_data['completion_time'], nn_predictions[:, 0], 
               alpha=0.5, label='Neural Network (with Robot Density)', color='magenta')
    
    # Add perfect prediction line
    min_val = min(test_data['completion_time'].min(), 
                 rf_predictions_basic[:, 0].min(),
                 nn_predictions_basic[:, 0].min(),
                 rf_predictions[:, 0].min(),
                 nn_predictions[:, 0].min())
    max_val = max(test_data['completion_time'].max(),
                 rf_predictions_basic[:, 0].max(),
                 nn_predictions_basic[:, 0].max(),
                 rf_predictions[:, 0].max(),
                 nn_predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Actual Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Model Comparison: Impact of Robot Density Feature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/feature_impact_comparison.png')
    plt.close()
    
    print("\nResults saved to:")
    print("- plots/feature_impact_comparison.png")

if __name__ == "__main__":
    main() 