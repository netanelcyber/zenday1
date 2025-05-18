import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from flask import Flask, request, jsonify


# Fetch the top 500 food data from OpenFoodFacts
def fetch_food_data(limit=500):
    url = f"https://world.openfoodfacts.org/products.json?fields=product_name,energy_100g,fat_100g,sugars_100g,proteins_100g,fiber_100g&page_size={limit}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data['products'])
    else:
        print("Failed to fetch data.")
        return pd.DataFrame()


# Preprocess the data based on GDM nutritional rules
def preprocess_data(df):
    # Select relevant features and handle missing values
    df = df[['product_name', 'energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g']]
    df = df.dropna()

    # GDM Classification Rules
    def classify_gdm(row):
        if row['sugars_100g'] > 15 or row['fat_100g'] > 20:  # High sugar or high fat
            return 1  # High risk (GDM)
        elif row['fiber_100g'] > 5 and row['proteins_100g'] > 10:  # High fiber and protein
            return 0  # Low risk (GDM)
        else:
            return 1  # Otherwise, classify as high risk

    # Apply classification function
    df['GDM_Class'] = df.apply(classify_gdm, axis=1)

    # Features and target
    X = df[['energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g']].values
    y = df['GDM_Class'].values

    return X, y


# Build and train a deep neural network model
def build_dnn(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    return model


# Flask app for real-time predictions
#app = Flask(__name__)


#@app.route('/predict', methods=['GET','POST'])
def predict():
    data = request.get_json()
    features = np.array([data['energy_100g'], data['fat_100g'], data['sugars_100g'], data['proteins_100g'],
                         data['fiber_100g']]).reshape(1, -1)

    # Scale features using the same scaler used during training
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] > 0.5:
        return ({"prediction": "High risk (GDM)"})
    else:
        return ({"prediction": "Low risk (GDM)"})


if __name__ == "__main__":
    # Fetch and preprocess the top 500 food products data
    food_data = fetch_food_data(limit=500)
    X, y = preprocess_data(food_data)

    # Train and save the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_dnn(X_train, y_train)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    predict()

    # Start the Flask app
#    app.run(debug=True)
