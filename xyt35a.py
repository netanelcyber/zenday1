import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# פונקציה להבאת נתוני מזון מ-OpenFoodFacts
def fetch_food_data(limit=10000):  # Increased limit to 10000
    url = f"https://world.openfoodfacts.org/products.json?fields=product_name,energy_100g,fat_100g,sugars_100g,proteins_100g,fiber_100g&page_size={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return pd.DataFrame(data['products'])
    except requests.exceptions.RequestException as e:
        print(f"שגיאה בהבאת נתונים: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"שגיאה בפענוח JSON: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"שגיאה בגישה לנתונים ב-JSON: {e}")
        return pd.DataFrame()



# פונקציה לעיבוד מקדים של הנתונים
def preprocess_data(df):
    if df.empty:
        return None, None

    # בחר את הפיצ'רים הרלוונטיים והתמודד עם ערכים חסרים
    df = df[['product_name', 'energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g']]
    df = df.dropna()

    if df.empty:
        return None, None
    # כללי סיווג של GDM עם התאמה להמלצות תזונתיות לסכרת הריון
    def classify_gdm(row):
        # ההמלצות התזונתיות מותאמות לסכרת הריון:
        # - הגבלת סוכרים: פחות מ-130 גרם ליום (בערך 10 גרם ל-100 גרם)
        # - צריכת שומן מתונה: כ-30-40% מהקלוריות (בערך 10-13 גרם ל-100 גרם)
        # - דגש על חלבון: כ-20-30% מהקלוריות (בערך 15-20 גרם ל-100 גרם)
        # - צריכת סיבים גבוהה: לפחות 28 גרם ליום (בערך 2 גרם ל-100 גרם)
        if row['sugars_100g'] > 10 or row['fat_100g'] > 13:  # סוכר גבוה או שומן גבוה
            return 5/5.0  # סיכון גבוה מאוד (5)
        elif row['fiber_100g'] > 2 and row['proteins_100g'] > 15:  # סיבים וחלבון גבוהים
            return 1/5.0  # סיכון נמוך מאוד (1)
        elif row['sugars_100g'] > 8 or row['fat_100g'] > 10:
            return 4/5.0 # סיכון גבוה
        elif row['fiber_100g'] > 1.5 and row['proteins_100g'] > 12:
            return 2/5.0 # סיכון נמוך
        else:
            return 3/5.0  # סיכון בינוני (3)

    # החל את פונקציית הסיווג
    df['GDM_Class'] = df.apply(classify_gdm, axis=1)

    # פיצ'רים ויעד
    X = df[['energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g']].values
    y = df['GDM_Class'].values
    product_names = df['product_name'].values # Extract product names.

    return X, y, product_names # Return product names as well


# פונקציה לבנייה ואימון של רשת עצבית עמוקה
def build_dnn(X_train, y_train):
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Increased neurons
    model.add(Dropout(0.3))  # Added dropout layer
    model.add(Dense(64, activation='tanh'))  # Changed activation function
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='elu'))  # Changed activation function
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10000, batch_size=32) #Added verbose=0 to suppress training output
    return model



def predict(model, scaler, data, product_names):
    """
    Make a prediction for a single food item and return the risk category.

    Args:
        model: Trained Keras model.
        scaler:  StandardScaler fitted on training data.
        data (dict): A dictionary containing the food's nutritional information
            ('energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g').
        product_names: list of product names

    Returns:
        str: The risk category (1-5).
    """
    try:
        # בדוק את כל המפתחות הנדרשים.
        required_keys = ['energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g']
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing one or more required keys. Required keys are {required_keys}")

        features = np.array([data['energy_100g'], data['fat_100g'], data['sugars_100g'], data['proteins_100g'],
                                    data['fiber_100g']]).reshape(1, -1)

        # קנה מידה של פיצ'רים באמצעות אותו סקיילר ששימש במהלך האימון
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        # Convert probability to risk score 1-5
        risk_score = 1 if prediction[0][0] < 0.2 else \
                     2 if prediction[0][0] < 0.4 else \
                     3 if prediction[0][0] < 0.6 else \
                     4 if prediction[0][0] < 0.8 else 5

        result = f"סיכון {risk_score} מתוך 5"
        return result

    except Exception as e:
        return f"Error: {e}"



if __name__ == "__main__":
    # הבא וpreprocess את 10000 נתוני מוצרי המזון המובילים
    food_data = fetch_food_data(limit=10000) # Get 10000 products
    X, y, product_names = preprocess_data(food_data) # Get product names.

    if X is not None and y is not None: # בדוק אם preprocess_data החזיר נתונים תקינים
        # אימון ושמירה של המודל
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # קנה מידה של פיצ'רים
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_dnn(X_train, y_train)

        # הערכת המודל
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        #  Make predictions on the entire dataset
        predictions = model.predict(scaler.transform(X))  # Scale the features before predicting
        risk_scores = [p[0]    for p in predictions]

        # Create a DataFrame with product names and predictions
        results_df = pd.DataFrame({'product_name': product_names, 'GDM_Risk': risk_scores})

        # Sort the DataFrame by the 'GDM_Class' column (0 for low risk, 1 for high risk)
        results_df = results_df.sort_values(by='GDM_Risk', ascending=False) # Sort by GDM_Class

        # Print the results.  Limit to a reasonable number of products.
        print(results_df.head(50).to_string()) # Print the top 50 products

        # Example of how to use the predict function:
        sample_data = {'energy_100g': 500, 'fat_100g': 30, 'sugars_100g': 20, 'proteins_100g': 5, 'fiber_100g': 2}
        prediction_result = predict(model, scaler, sample_data, product_names)
        print(f"Prediction for sample data: {prediction_result}")
    else:
        print("Failed to get data, cannot train model and make predictions")
