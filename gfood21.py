import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, request, jsonify
import os  # Import the os module


# פונקציה להבאת נתוני מזון מ-OpenFoodFacts
def fetch_food_data(limit=500):
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
            return 1  # סיכון גבוה (GDM)
        elif row['fiber_100g'] > 2 and row['proteins_100g'] > 15:  # סיבים וחלבון גבוהים
            return 0  # סיכון נמוך (GDM)
        else:
            return 1  # אחרת, סווג כסיכון גבוה

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
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0) #Added verbose=0 to suppress training output

    return model



app = Flask(__name__)  # אתחול אפליקציית Flask


@app.route('/predict', methods=['POST']) # שונה ל-POST, מכיוון ש-GET אינו מתאים לשליחת נתונים בגוף הבקשה
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "לא סופקו נתונים"}), 400

        # בדוק את כל המפתחות הנדרשים.
        required_keys = ['energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'fiber_100g']
        if not all(key in data for key in required_keys):
            return jsonify({"error": "חסר אחד או יותר מהמפתחות הנדרשים. המפתחות הנדרשים הם energy_100g, fat_100g, sugars_100g, proteins_100g, fiber_100g"}), 400

        features = np.array([data['energy_100g'], data['fat_100g'], data['sugars_100g'], data['proteins_100g'],
                                    data['fiber_100g']]).reshape(1, -1)

        # קנה מידה של פיצ'רים באמצעות אותו סקיילר ששימש במהלך האימון
        features_scaled = scaler.transform(features) # השתמש בסקיילר שאומן על נתוני האימון

        prediction = model.predict(features_scaled)

        if prediction[0][0] > 0.5: # גש לערך בתוך הרשימה המקוננת.
            result = "סיכון גבוה (GDM)"
        else:
            result = "סיכון נמוך (GDM)"

        # Return the prediction and product name
        return jsonify({"prediction": result, "product_name": product_names[0]}) # Return the first product name.  You might need to adjust this.

    except Exception as e:
        return jsonify({"error": f"אירעה שגיאה: {e}"}), 500



if __name__ == "__main__":
    # הבא וpreprocess את 500 נתוני מוצרי המזון המובילים
    food_data = fetch_food_data(limit=500)
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
        print(f"דיוק המודל: {accuracy * 100:.2f}%")

        # הפעל את אפליקציית Flask. השתמש ביציאה פנויה.
        app.run(debug=False, port=int(os.environ.get('PORT', 5000)))
    else:
        print("נכשל בהשגת נתונים, לא ניתן לאמן את המודל ולהפעיל את השרת")
