from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import pickle
import os

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def train_model(self, data_path):
        """Train the SVM model with the provided dataset"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The data file at {data_path} does not exist.")
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=56)
        
        # Train the SVM model
        print("Training the SVM model...")
        self.model = SVC(C=1, kernel='linear', probability=True)
        self.model.fit(X_train, y_train)
        print("Model training completed.")
        
        # Save the model
        model_dir = "src/models"
        os.makedirs(model_dir, exist_ok=True)
        with open(f"{model_dir}/svm_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved successfully.")
    
    #def load_model(self):

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model("data/scaled_data.csv")
    print("Model training completed.")  # This line is added to the original script