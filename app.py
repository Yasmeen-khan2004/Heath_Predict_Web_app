from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
random_forest = joblib.load('./models/randomforest.lb')

# Load and preprocess the dataset
dataset_path = 'Disease_symptom_and_patient_profile_dataset.csv' 
df = pd.read_csv(dataset_path)
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

app = Flask(__name__)

# List of disease names
disease_names = df['Disease'].unique()
disease_names = label_encoder.inverse_transform(disease_names)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html', disease_names=disease_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Define the expected features
    features = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty_Breathing', 'Age', 'Gender', 'Blood_Pressure', 'Cholesterol_Level']
    
    # Retrieve form data
    user_input = []
    for feature in features:
        value = request.form.get(feature)
        if value is None:
            return "Error: Missing form data", 400
        user_input.append(value)
    
    # Map disease name to encoded integer
    disease_name = user_input[0]
    try:
        disease_encoded = label_encoder.transform([disease_name])[0]
    except ValueError:
        return "Error: Disease name not recognized", 400
    
    # Update user input with encoded disease
    user_input[0] = disease_encoded
    
    # Convert user input to numeric (assuming it's string input)
    try:
        user_input = list(map(float, user_input))
    except ValueError:
        return "Error: Invalid data format", 400
    
    # Debug print
    print(f"Processed user input: {user_input}")

    # Predict using the model
    try:
        prediction = random_forest.predict([user_input])[0]
    except Exception as e:
        return f"Error: Model prediction failed - {e}", 500
    
    # Determine the prediction label
    result = 'Positive' if prediction == 1 else 'Negative'
    
    return render_template('project.html', disease_names=disease_names, prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
