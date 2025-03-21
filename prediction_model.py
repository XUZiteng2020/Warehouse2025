import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Enable Metal backend
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def pad_layout(layout, target_height, target_width):
    """Pad layout to target size"""
    current_height, current_width = layout.shape
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)
    
    return np.pad(
        layout,
        ((0, pad_height), (0, pad_width)),
        mode='constant',
        constant_values=0
    )

class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, features, targets):
        """Train the random forest model"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, targets)
        
    def predict(self, features):
        """Make predictions"""
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def evaluate(self, features, targets):
        """Evaluate model performance"""
        predictions = self.predict(features)
        mae = np.mean(np.abs(predictions - targets), axis=0)
        return {
            'completion_time_mae': mae[0],
            'orders_per_step_mae': mae[1],
            'robot_efficiency_mae': mae[2],
            'completion_rate_mae': mae[3]
        }

class WarehousePredictor:
    def __init__(self, data_path='warehouse_data_files'):
        self.data_path = data_path
        self.layout_path = os.path.join(data_path, 'layouts')
        self.csv_path = os.path.join(data_path, 'training_data.csv')
        
        # Model components
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        # Load CSV data
        self.df = pd.read_csv(self.csv_path)
        
        # Load layout data
        layouts = []
        print("Loading layout files...")
        
        # First pass to find maximum dimensions
        max_height = 0
        max_width = 0
        for idx in tqdm(range(len(self.df)), desc="Finding max dimensions"):
            layout_file = f'layout_{idx:04d}.npy'
            layout = np.load(os.path.join(self.layout_path, layout_file))
            max_height = max(max_height, layout.shape[0])
            max_width = max(max_width, layout.shape[1])
        
        # Second pass to load and pad layouts
        print("\nPadding layouts...")
        for idx in tqdm(range(len(self.df)), desc="Loading and padding"):
            layout_file = f'layout_{idx:04d}.npy'
            layout = np.load(os.path.join(self.layout_path, layout_file))
            padded_layout = pad_layout(layout, max_height, max_width)
            layouts.append(padded_layout)
        
        layouts = np.array(layouts)
        
        # Normalize layouts to be between 0 and 1
        layouts = layouts / 3.0  # Since we have values 0,1,2,3
        
        # Extract features and targets
        self.features_numeric = self.df[[
            'warehouse_width', 'warehouse_height', 'num_robots',
            'num_workstations', 'order_density', 'shelf_count'
        ]].values
        
        self.targets = self.df[[
            'completion_time', 'orders_per_step', 
            'robot_efficiency', 'completion_rate'
        ]].values
        
        # Split the data
        indices = np.arange(len(self.df))
        (train_idx, test_idx, 
         self.train_layouts, self.test_layouts,
         self.train_features, self.test_features,
         self.train_targets, self.test_targets) = train_test_split(
            indices, layouts, self.features_numeric, self.targets, 
            test_size=0.2, random_state=42
        )
        
        # Scale numeric features
        self.scaler.fit(self.train_features)
        self.train_features_scaled = self.scaler.transform(self.train_features)
        self.test_features_scaled = self.scaler.transform(self.test_features)
        
        print(f"\nTraining samples: {len(self.train_layouts)}")
        print(f"Test samples: {len(self.test_layouts)}")
        print(f"Layout shape: {self.train_layouts[0].shape}")
        
    def build_model(self):
        """Build the hybrid CNN + numeric features model"""
        # Input for layout images
        layout_input = Input(shape=self.train_layouts[0].shape + (1,))
        
        # CNN for layout processing
        x = Conv2D(32, (3, 3), activation='relu')(layout_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        layout_features = Flatten()(x)
        
        # Input for numeric features
        numeric_input = Input(shape=(self.train_features.shape[1],))
        
        # Combine features
        combined = Concatenate()([layout_features, numeric_input])
        
        # Dense layers for prediction
        x = Dense(128, activation='relu')(combined)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(4, activation='linear')(x)  # 4 targets
        
        # Create model
        self.model = Model(
            inputs=[layout_input, numeric_input],
            outputs=outputs
        )
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        # Reshape layouts for CNN
        train_layouts_reshaped = self.train_layouts.reshape(
            self.train_layouts.shape + (1,)
        )
        test_layouts_reshaped = self.test_layouts.reshape(
            self.test_layouts.shape + (1,)
        )
        
        # Train the model
        history = self.model.fit(
            [train_layouts_reshaped, self.train_features_scaled],
            self.train_targets,
            validation_data=(
                [test_layouts_reshaped, self.test_features_scaled],
                self.test_targets
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def predict(self, layout, features):
        """Make predictions for new warehouse configurations"""
        # Preprocess inputs
        layout_reshaped = layout.reshape(1, *layout.shape, 1)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(
            [layout_reshaped, features_scaled],
            verbose=0
        )
        
        return {
            'completion_time': prediction[0][0],
            'orders_per_step': prediction[0][1],
            'robot_efficiency': prediction[0][2],
            'completion_rate': prediction[0][3]
        }
    
    def evaluate(self):
        """Evaluate model performance on test set"""
        # Reshape layouts for CNN
        test_layouts_reshaped = self.test_layouts.reshape(
            self.test_layouts.shape + (1,)
        )
        
        # Get predictions
        predictions = self.model.predict(
            [test_layouts_reshaped, self.test_features_scaled],
            verbose=0
        )
        
        # Calculate metrics
        metrics = {
            'completion_time_mae': np.mean(np.abs(predictions[:, 0] - self.test_targets[:, 0])),
            'orders_per_step_mae': np.mean(np.abs(predictions[:, 1] - self.test_targets[:, 1])),
            'robot_efficiency_mae': np.mean(np.abs(predictions[:, 2] - self.test_targets[:, 2])),
            'completion_rate_mae': np.mean(np.abs(predictions[:, 3] - self.test_targets[:, 3]))
        }
        
        return metrics

def plot_completion_time_comparison(ground_truth, cnn_pred, rf_pred):
    """Plot completion time comparison between ground truth and predictions"""
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot
    plt.scatter(ground_truth, cnn_pred, alpha=0.5, label='CNN Predictions')
    plt.scatter(ground_truth, rf_pred, alpha=0.5, label='Random Forest Predictions')
    
    # Add perfect prediction line
    min_val = min(ground_truth.min(), cnn_pred.min(), rf_pred.min())
    max_val = max(ground_truth.max(), cnn_pred.max(), rf_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Completion Time')
    plt.ylabel('Predicted Completion Time')
    plt.title('Completion Time: Ground Truth vs Predictions')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/completion_time_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Create and train the CNN model
    print("Training CNN model...")
    cnn_predictor = WarehousePredictor()
    cnn_predictor.load_data()
    cnn_predictor.build_model()
    cnn_history = cnn_predictor.train(epochs=50, batch_size=32)
    
    # Create and train the Random Forest model
    print("\nTraining Random Forest model...")
    rf_predictor = RandomForestPredictor()
    rf_predictor.train(cnn_predictor.train_features, cnn_predictor.train_targets)
    
    # Get predictions from both models
    # CNN predictions
    test_layouts_reshaped = cnn_predictor.test_layouts.reshape(
        cnn_predictor.test_layouts.shape + (1,)
    )
    cnn_predictions = cnn_predictor.model.predict(
        [test_layouts_reshaped, cnn_predictor.test_features_scaled],
        verbose=0
    )
    
    # RF predictions
    rf_predictions = rf_predictor.predict(cnn_predictor.test_features)
    
    # Evaluate both models
    print("\nCNN Model Performance:")
    cnn_metrics = {
        'completion_time_mae': np.mean(np.abs(cnn_predictions[:, 0] - cnn_predictor.test_targets[:, 0])),
        'orders_per_step_mae': np.mean(np.abs(cnn_predictions[:, 1] - cnn_predictor.test_targets[:, 1])),
        'robot_efficiency_mae': np.mean(np.abs(cnn_predictions[:, 2] - cnn_predictor.test_targets[:, 2])),
        'completion_rate_mae': np.mean(np.abs(cnn_predictions[:, 3] - cnn_predictor.test_targets[:, 3]))
    }
    for metric, value in cnn_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nRandom Forest Model Performance:")
    rf_metrics = rf_predictor.evaluate(cnn_predictor.test_features, cnn_predictor.test_targets)
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot completion time comparison
    plot_completion_time_comparison(
        cnn_predictor.test_targets[:, 0],  # Ground truth
        cnn_predictions[:, 0],             # CNN predictions
        rf_predictions[:, 0]               # RF predictions
    )
    print("\nComparison plot saved as 'plots/completion_time_comparison.png'") 