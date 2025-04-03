import os
from flask import Flask, request, render_template, jsonify

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)

# Load and train the model
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, 0:13]
y = dataset.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Train the model
classifier = KNeighborsClassifier(n_neighbors=32, p=2, metric='euclidean')
classifier.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create input array for prediction
        input_data = [[
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]]
        
        # Scale the input data
        input_scaled = sc_X.transform(input_data)
        
        # Make prediction
        prediction = classifier.predict(input_scaled)
        
        # Return result
        return jsonify({
            'prediction': int(prediction[0]),
            'message': 'You may have heart disease.' if prediction[0] == 1 else 'You are likely healthy.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 