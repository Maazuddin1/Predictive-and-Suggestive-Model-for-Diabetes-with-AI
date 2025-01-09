import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, filepath):
        """Load and return the dataset"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} does not exist.")
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df

    def preprocess_data(self, df):
        """Preprocess the data by handling missing values"""
        # Handle missing values (zeros)
        features_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
        for feature in features_to_process:
            mean_value = df[feature].replace(0, np.nan).mean()
            df[feature] = df[feature].replace(0, mean_value)
        
        print("Missing values handled.")
        return df

    def split_data(self, df):
        """Split data into features and target"""
        features = df.drop('Outcome', axis=1)
        target = df['Outcome']
        return features, target

    def scale_features(self, features, is_training=False):
        """Scale features using StandardScaler"""
        if is_training:
            scaled_features = self.scaler.fit_transform(features)
            # Save the scaler for future use
            model_dir = "src/models"
            os.makedirs(model_dir, exist_ok=True)
            # Save the scaler as a pickle file
            with open(f"{model_dir}/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            # Save the scaled data as a CSV file
            scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
            scaled_df['Outcome'] = df['Outcome']  # Add the Outcome column back
            scaled_csv_path = "data/scaled_data.csv"
            scaled_df.to_csv(scaled_csv_path, index=False)
            print("Scaled data saved as csv file.")
        else:
            scaled_features = self.scaler.transform(features)
        
        return scaled_features


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data("data/preprocessed_data.csv")
    df = preprocessor.preprocess_data(df)

    # Split data into features and target
    features, target = preprocessor.split_data(df)

    # Scale features (Training phase)
    scaled_features = preprocessor.scale_features(features, is_training=True)
    
    print("Data preprocessing completed.")
