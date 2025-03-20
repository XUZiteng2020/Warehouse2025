import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
import os
from tqdm import tqdm

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
        df = pd.read_csv(self.csv_path)
        
        # Load layout data
        layouts = []
        print("Loading layout files...")
        
        # First pass to find maximum dimensions
        max_height = 0
        max_width = 0
        for idx in tqdm(range(len(df)), desc="Finding max dimensions"):
            layout_file = f'layout_{idx:04d}.npy'
            layout = np.load(os.path.join(self.layout_path, layout_file))
            max_height = max(max_height, layout.shape[0])
            max_width = max(max_width, layout.shape[1])
        
        # Second pass to load and pad layouts
        print("\nPadding layouts...")
        for idx in tqdm(range(len(df)), desc="Loading and padding"):
            layout_file = f'layout_{idx:04d}.npy'
            layout = np.load(os.path.join(self.layout_path, layout_file))
            padded_layout = pad_layout(layout, max_height, max_width)
            layouts.append(padded_layout)
        
        layouts = np.array(layouts)
        
        # Normalize layouts to be between 0 and 1
        layouts = layouts / 3.0  # Since we have values 0,1,2,3
        
        # Extract features and targets
        features_numeric = df[[
            'warehouse_width', 'warehouse_height', 'num_robots',
            'num_workstations', 'order_density', 'shelf_count'
        ]].values
        
        targets = df[[
            'completion_time', 'orders_per_step', 
            'robot_efficiency', 'completion_rate'
        ]].values
        
        # Split the data
        indices = np.arange(len(df))
        (train_idx, test_idx, 
         self.train_layouts, self.test_layouts,
         self.train_features, self.test_features,
         self.train_targets, self.test_targets) = train_test_split(
            indices, layouts, features_numeric, targets, 
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

if __name__ == "__main__":
    # Create and train the model
    predictor = WarehousePredictor()
    predictor.load_data()
    predictor.build_model()
    
    # Train the model
    history = predictor.train(epochs=50, batch_size=32)
    
    # Evaluate performance
    metrics = predictor.evaluate()
    print("\nTest Set Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Example prediction
    sample_layout = predictor.test_layouts[0]
    sample_features = predictor.test_features[0]
    prediction = predictor.predict(sample_layout, sample_features)
    print("\nSample Prediction:")
    for metric, value in prediction.items():
        print(f"{metric}: {value:.4f}") 