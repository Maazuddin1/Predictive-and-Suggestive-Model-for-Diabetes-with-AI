import pandas as pd
import pickle

class PredictionModel:
    def __init__(self, model_path, scaler_path):
        # Load the trained model
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        
        # Load the scaler
        with open(scaler_path, 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)

    def predict(self, features, feature_columns):
        try:
            # Convert features to DataFrame with proper column names
            features_df = pd.DataFrame([features], columns=feature_columns)
            # Scale features using the scaler
            scaled_features = self.scaler.transform(features_df)  # Make sure features_df has the correct column names
            scaled_features = pd.DataFrame(scaled_features, columns=feature_columns)

            # Make predictions
            prediction = self.model.predict(scaled_features)
            probability = self.model.predict_proba(scaled_features)[0][1]
            
            return {
                "prediction": int(prediction[0]),
                "probability": probability
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

# Example usage
if __name__ == "__main__":
    model_path = "src/models/svm_model.pkl"  # Path to the trained model
    scaler_path = "src/models/scaler.pkl"    # Path to the saved scaler
    
    # Example input features (they should match the training data structure)
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    input_features = [6,148.0,72.0,35.0,155.5482233502538,33.6,0.627,50]  # Replace with actual input
    
    predictor = PredictionModel(model_path, scaler_path)
    result = predictor.predict(input_features, feature_columns)
    
    if result:
        print(f"Prediction: {result['prediction']}, Probability: {result['probability']}")
