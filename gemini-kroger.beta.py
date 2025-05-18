import sys
import json
import requests
import numpy as np
import faiss
from collections import defaultdict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout,
    QWidget, QFileDialog, QInputDialog, QMessageBox
)
from sentence_transformers import SentenceTransformer
import google.generativeai as genai




# === CONFIG ===
#genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your actual Gemini API key
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === Kroger OAuth2 Token ===
#C = "Basic YOUR_BASE64_CREDENTIALS"  # Replace with actual Kroger credentials









import sys
import json
import requests
import numpy as np
import faiss
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout,
    QWidget, QFileDialog, QInputDialog, QMessageBox
)
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === CONFIG ===
genai.configure(api_key="AIzaSyDgqFUJYnozGy3TmJwMKw_LpsdJtWHRnKM")  # Replace with your Gemini key
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === Kroger OAuth2 Token ===
CREDS = 'Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OT='
C="Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4Om40Wm50Q0ZobEdqdHpLY0JKcVI2VUx3QUVTcTM2TGx6R1NFbkJvOWg="


def get_kroger_token():
    try:
        res = requests.post(
            url="https://api.kroger.com/v1/connect/oauth2/token",
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                "Authorization": C
            },
            data={"grant_type": "client_credentials", "scope": "product.compact"}
        )
        res.raise_for_status()
        token = res.json().get("access_token", "")
        if not token:
            print("❌ Token missing from response:", res.json())
        return token
    except Exception as e:
        print("❌ Kroger Token Error:", str(e))
        return ""





def get_kro2ge2r_token():
    try:
        res = requests.post(
            url="https://api.kroger.com/v1/connect/oauth2/token",
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                "Authorization": C
            },
            data={"grant_type": "client_credentials", "scope": "product.compact"}
        )
        res.raise_for_status()
        token = res.json().get("access_token", "")
        if not token:
            print("❌ Token missing from response:", res.json())
        return token
    except Exception as e:
        print("❌ Kroger Token Error:", str(e))
        return ""

class MealPlannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDM Meal Planner (PyQt6)")
        self.setMinimumSize(1000, 700)

        self.kroger_token = get_kroger_token()

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        btn_upload = QPushButton("\ud83d\udcc2 Upload Kroger JSON")
        btn_search = QPushButton("\ud83d\udd0d Semantic Search")
        btn_generate = QPushButton("\ud83c\udf7d\ufe0f Generate Meal Plan")
        btn_kroger_zip = QPushButton("\ud83d\uded2 Search Kroger by ZIP")

        btn_upload.clicked.connect(self.upload_json)
        btn_search.clicked.connect(self.semantic_search)
        btn_generate.clicked.connect(self.generate_meal_plan)
        btn_kroger_zip.clicked.connect(self.kroger_zip_search)

        layout = QVBoxLayout()
        layout.addWidget(btn_upload)
        layout.addWidget(btn_search)
        layout.addWidget(btn_generate)
        layout.addWidget(btn_kroger_zip)
        layout.addWidget(self.output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def log(self, text):
        self.output.append(text)

    def upload_json(self):
        global index_is_empty
        files, _ = QFileDialog.getOpenFileNames(self, "Upload JSON", "", "JSON Files (*.json)")
        if not files:
            return

        all_desc = []
        new_meta = []

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("data", []):
                        desc = item.get("description", "").strip()
                        if desc:
                            all_desc.append(desc)
                            new_meta.append({
                                "productId": item.get("productId"),
                                "brand": item.get("brand"),
                                "description": desc,
                                "category": ", ".join(item.get("categories", [])),
                                "image": next((img["sizes"][0]["url"]
                                               for img in item.get("images", []) if img.get("sizes")), ""),
                                "link": f"https://www.kroger.com/p/{desc.replace(' ', '-').lower()}/{item.get('productId')}"
                            })
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load {path}:\n{str(e)}")
                return

        if all_desc:
            vectors = embedding_model.encode(all_desc)
            index.add(np.array(vectors).astype("float32"))
            product_metadata.extend(new_meta)
            index_is_empty = False
            self.log(f"✅ Indexed {len(new_meta)} products.")
        else:
            self.log("⚠️ No valid descriptions found.")

    def semantic_search(self):
        if index_is_empty:
            QMessageBox.warning(self, "Error", "Upload products first.")
            return
        query, ok = QInputDialog.getText(self, "Search", "Enter product concept:")
        if not ok or not query:
            return

        qvec = embedding_model.encode([query])
        D, I = index.search(np.array(qvec).astype("float32"), k=5)
        results = [product_metadata[i] for i in I[0] if i < len(product_metadata)]

        self.log(f"\n🔍 Results for '{query}':\n")
        for p in results:
            self.log(f"- {p['description']} ({p['brand']})\n  Link: {p['link']}\n")

    def generate_meal_plan(self):
        if not product_metadata:
            QMessageBox.warning(self, "Missing Data", "Upload products first.")
            return
        goal, ok = QInputDialog.getText(self, "Meal Goal", "Describe your meal goal:")
        if not ok or not goal:
            return

        grouped = defaultdict(list)
        for p in product_metadata:
            key = p.get("category") or p.get("brand") or "Other"
            grouped[key].append(p)

        sampled_products = []
        for group in list(grouped.values())[:8]:
            sampled_products.append(group[0])

        ingredients_json = json.dumps(sampled_products, indent=2)

        prompt = f"""
You are a licensed nutritionist designing a meal plan for a woman with Gestational Diabetes Mellitus (GDM).

She has the following ingredients available in her kitchen:
{ingredients_json}

Her goal is:
"{goal}"

Please generate a JSON-formatted 1- to 3-day meal plan that includes:
- 3 meals and 2 snacks per day
- Ingredient lists (with product links)
- Portion sizes appropriate for blood sugar control
- Calories, Carbs, Protein, and Fat per meal
- A short explanation of how each meal supports GDM dietary needs
- A daily nutrition summary + practical tips for glycemic balance

Include all product links using their \"link\" field.
"""
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            result = model.generate_content(prompt)
            text = result.text.strip()

            try:
                meal_plan = json.loads(text)
            except json.JSONDecodeError:
                meal_plan = {"raw_text": text}

            self.log("\n🍽️ Improved Meal Plan:\n")
            self.log(json.dumps(meal_plan, indent=2) if isinstance(meal_plan, dict) else text)

        except Exception as e:
            QMessageBox.critical(self, "Gemini Error", str(e))

    def kroger_zip_search(self):
        if not self.kroger_token:
            QMessageBox.warning(self, "Token", "Missing Kroger token.")
            return
        zip_code, ok1 = QInputDialog.getText(self, "ZIP", "Enter ZIP Code:")
        term, ok2 = QInputDialog.getText(self, "Query", "Product name:")
        if not (ok1 and ok2 and zip_code and term):
            return

        headers = {"Authorization": f"Bearer {self.kroger_token}"}
        try:
            locs = requests.get("https://api.kroger.com/v1/locations", headers=headers, params={
                "filter.zipCode.near": zip_code,
                "filter.radiusInMiles": 30,
                "filter.chain": "Kroger"
            }).json().get("data", [])

            if not locs:
                self.log("No Kroger stores found.")
                return

            for store in locs[:1]:
                loc_id = store.get("locationId")
                addr = store.get("address", {}).get("addressLine1", "Unknown")
                products = requests.get("https://api.kroger.com/v1/products", headers=headers, params={
                    "filter.term": term,
                    "filter.locationId": loc_id,
                    "filter.limit": 10
                }).json().get("data", [])

                self.log(f"\n🛒 {addr} | Search: {term}")
                for p in products:
                    price = p.get("items", [{}])[0].get("price", {}).get("regular", "?")
                    link = f"https://www.kroger.com/p/{p['description'].replace(' ', '-').lower()}/{p['productId']}"
                    self.log(f"- {p['description']} ({p['brand']}) - ${price}\n  {link}")
        except Exception as e:
            QMessageBox.critical(self, "Kroger Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MealPlannerApp()
    window.show()
    sys.exit(app.exec())
