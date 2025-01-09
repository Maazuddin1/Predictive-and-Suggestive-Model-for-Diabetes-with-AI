from typing import Dict, List
import os
import google.generativeai as genai
import re

class HealthRecommendations:
    def __init__(self):
        # Configure Google GenAI
        api_key = os.getenv('AIzaSyAdCzdbvGIWIDfVdZre1n-10SIBf9bgcVk')  # Fetch from environment variable
        genai.configure(api_key='AIzaSyAdCzdbvGIWIDfVdZre1n-10SIBf9bgcVk')

    def get_recommendations(self, patient_data: Dict, prediction: Dict) -> Dict[str, List[str]]:
        """Generate personalized health recommendations based on patient data and prediction"""

        # Create a prompt for the LLM
        prompt = self._create_prompt(patient_data, prediction)

        try:
            print("Generated prompt:", prompt)  # Debugging: Check the prompt

            model = genai.GenerativeModel("gemini-1.5-flash")
            print("Model initialized successfully")  # Debugging

            response = model.generate_content(
                contents=[{"parts": [{"text": prompt}]}],
                generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=500)
            )

            print("Response from API:", response)  # Debugging: Check the response

            if response.candidates and response.candidates[0].content.parts: # Check if response and parts exist
              response_text = response.candidates[0].content.parts[0].text
              recommendations = self._parse_recommendations(response_text)
              return recommendations
            else:
              print("Unexpected response format from the API.")
              return self._get_fallback_recommendations(prediction['is_diabetic'])

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return self._get_fallback_recommendations(prediction['is_diabetic'])
        
    def _create_prompt(self, patient_data: Dict, prediction: Dict) -> str:
        """Create a prompt for the LLM based on patient data"""
        risk_level = "high" if prediction['probability'] > 0.7 else "moderate" if prediction['probability'] > 0.3 else "low"

        prompt = f"""
        Based on the following patient data:
        - Glucose Level: {patient_data['Glucose']}
        - Blood Pressure: {patient_data['BloodPressure']}
        - BMI: {patient_data['BMI']}
        - Age: {patient_data['Age']}
        - Diabetes Risk Level: {risk_level}

        Please provide specific recommendations in the following categories:
        1. Diet and Nutrition
        2. Physical Activity
        3. Lifestyle Changes
        4. Monitoring and Prevention

        Make the recommendations specific to this patient's condition and risk level.
        """
        return prompt

    def _parse_recommendations(self, response: str) -> Dict[str, List[str]]:
        """Parse the LLM response into structured recommendations and remove all asterisks."""
        
        # Categories we expect in the response
        categories = ['Diet and Nutrition', 'Physical Activity', 'Lifestyle Changes', 'Monitoring and Prevention']
        recommendations = {category: [] for category in categories}

        # Assuming the text is extracted from the API response
        api_response_text = response  # This would be the text from the API, adjust based on the actual response structure

        # Regex patterns to match categories and extract their associated recommendations
        current_category = None
        lines = api_response_text.split("\n")

        for line in lines:
            line = line.strip()

            # Check if the line is a category
            if any(category in line for category in categories):
                for category in categories:
                    if category in line:
                        current_category = category
                        break
            elif line and current_category:
                # Remove all asterisks from the line
                cleaned_line = re.sub(r'\*+', '', line).strip()  # Remove all asterisks and leading/trailing spaces
                if cleaned_line:  # Add only non-empty lines
                    recommendations[current_category].append(cleaned_line)

        return recommendations

    def _get_fallback_recommendations(self, is_diabetic: bool) -> Dict[str, List[str]]:
        """Provide fallback recommendations if API call fails"""
        if is_diabetic:
            return {
                '1.Diet and Nutrition': [
                    'Monitor carbohydrate intake and follow a balanced diet',
                    'Eat plenty of vegetables and whole grains',
                    'Limit sugary foods and beverages'
                ],
                'Physical Activity': [
                    'Aim for 150 minutes of moderate exercise per week',
                    'Include both aerobic and strength training exercises',
                    'Take regular walking breaks during the day'
                ],
                'Lifestyle Changes': [
                    'Monitor blood sugar regularly',
                    'Maintain a healthy sleep schedule',
                    'Manage stress through relaxation techniques'
                ],
                'Monitoring and Prevention': [
                    'Regular check-ups with healthcare provider',
                    'Keep track of blood sugar levels',
                    'Monitor blood pressure and weight.1'
                ]
            }
        else:
            return {
                '2.Diet and Nutrition': [
                    'Follow a balanced diet rich in whole foods',
                    'Limit processed foods and added sugars',
                    'Stay hydrated with water'
                ],
                'Physical Activity': [
                    'Regular exercise for 30 minutes daily',
                    'Include variety in your workout routine',
                    'Stay active throughout the day'
                ],
                'Lifestyle Changes': [
                    'Maintain a healthy weight',
                    'Get adequate sleep',
                    'Practice stress management'
                ],
                'Monitoring and Prevention': [
                    'Regular health check-ups',
                    'Annual blood sugar screening',
                    'Monitor weight and blood pressure.2'
                ]
            }

