from flask import Flask, request, render_template
from src.scripts.prediction import PredictionModel
from src.scripts.health_recommendations import HealthRecommendations

app = Flask(__name__, template_folder='../templates')

# Define paths for model and scaler based on your directory structure
MODEL_PATH = "src/models/svm_model.pkl"
SCALER_PATH = "src/models/scaler.pkl"

# Define feature columns
FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Initialize components
try:
    predictor = PredictionModel(MODEL_PATH, SCALER_PATH)
    health_advisor = HealthRecommendations()
except Exception as e:
    print(f"Error initializing components: {e}")
    raise


@app.route('/', methods=['GET', 'POST'])
def home():
    # For GET requests, render an empty form
    if request.method == 'GET':
        return render_template('index.html')

    # For POST requests, handle the form submission
    elif request.method == 'POST':
        try:
            # Collect form data
            features = []
            form_data = {}

            for field in FEATURE_COLUMNS:
                value = request.form.get(field)
                if not value:
                    raise ValueError(f"Missing required field: {field}")
                features.append(float(value))
                form_data[field] = value

            # Generate prediction
            prediction_result = predictor.predict(features, FEATURE_COLUMNS)
            if not prediction_result:
                raise ValueError("Failed to generate prediction")

            # Format prediction for health recommendations
            health_prediction = {
                'is_diabetic': bool(prediction_result['prediction']),
                'probability': prediction_result['probability']
            }

            # Get health recommendations from Gemini API
            recommendations = health_advisor.get_recommendations(
                patient_data={
                    'Glucose': form_data['Glucose'],
                    'BloodPressure': form_data['BloodPressure'],
                    'BMI': form_data['BMI'],
                    'Age': form_data['Age']
                },
                prediction=health_prediction
            )

            # Prepare result message
            if health_prediction['is_diabetic']:
                result = f"Based on the analysis, this person is likely diabetic (Confidence: {health_prediction['probability']*100:.1f}%)"
            else:
                result = f"Based on the analysis, this person is not likely diabetic (Confidence: {(1-health_prediction['probability'])*100:.1f}%)"

            # Render the page with results
            return render_template(
                'index.html',
                result=result,
                suggestions=recommendations,
                input_data=form_data
            )

        except ValueError as ve:
            # Handle missing or invalid data
            return render_template('index.html', error=str(ve))
        except Exception as e:
            print(f"Error in prediction or recommendation generation: {e}")
            return render_template('index.html', error="An unexpected error occurred during prediction.")


@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error occurred."), 500


if __name__ == '__main__':
    app.run(debug=True)
