import os
import json
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template_string
#from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === CONFIG ===
genai.configure(api_key="")  # ← Replace with yours
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # Vector size for this model
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === FLASK APP ===
app = Flask(__name__)
os.makedirs("uploaded_jsons", exist_ok=True)

@app.route("/")
def index5():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>GDM Meal Planner</title></head>
    <body>
        <h2>Upload Kroger Product JSON</h2>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" multiple required>
            <button type="submit">Upload</button>
        </form>
        <h3>Search for Product</h3>
    <form method="post" action="/search">
    <input type="text" name="query" placeholder="e.g. tofu" required>
    <input type="text" name="zip" placeholder="Enter your ZIP code" pattern="\d{5}" title="Five digit ZIP code" required>
    <button type="submit">Search</button>
</form>

<h3>Semantic Search Products</h3>
<form method="post" action="/search2">
    <input type="text" name="query" placeholder="e.g. healthy snacks" required>
    <button type="submit">Semantic Search</button>
</form>



        <h3>Generate GDM Meal Plan</h3>
        <form method="post" action="/mealplan">
            <input type="text" name="goal" placeholder="e.g. 3-day GDM-friendly meal plan with snacks" required>
            <button type="submit">Generate</button>
        </form>
    </body>
    </html>
    """)



@app.route("/search", methods=["POST"])
def search_products_and_stores():
    query = request.form.get("query")
    zip_code = request.form.get("zip")

    if not query or not zip_code:
        return jsonify({"error": "Both search term and ZIP code are required."}), 400

    # Step 1: Fetch nearby stores
    headers = {"Authorization": f"Bearer {token}"}
    location_params = {
        "filter.zipCode.near": zip_code,
        "filter.radiusInMiles": "15",
        "filter.chain": "Kroger"
    }
    location_response = requests.get("https://api.kroger.com/v1/locations", headers=headers, params=location_params)

    if location_response.status_code != 200:
        return jsonify({"error": "Failed to fetch store locations."}), 500

    stores = location_response.json().get("data", [])
    if not stores:
        return jsonify({"error": "No stores found near the provided ZIP code."}), 404

    # Step 2: Search for products in each store
    results = []
    for store in stores:
        location_id = store.get("locationId")
        if not location_id:
            continue

        product_params = {
            "filter.term": query,
            "filter.locationId": location_id,
            "filter.limit": "5"
        }
        product_response = requests.get("https://api.kroger.com/v1/products", headers=headers, params=product_params)

        if product_response.status_code != 200:
            continue

        products = product_response.json().get("data", [])
        for product in products:
            results.append({
                "store": {
                    "locationId": location_id,
                    "address": store.get("address", {}).get("addressLine1", "Address not available")
                },
                "product": {
                    "description": product.get("description"),
                    "brand": product.get("brand"),
                    "price": product.get("items", [{}])[0].get("price", {}).get("regular"),
                    "image": product.get("images", [{}])[0].get("sizes", [{}])[0].get("url"),
                    "link": f"https://www.kroger.com/p/{product.get('description', '').replace(' ', '-').lower()}/{product.get('productId')}"
                }
            })

    if not results:
        return jsonify({"message": "No matching products found in nearby stores."}), 404

    return jsonify({"query": query, "results": results})

vectors = [np.random.rand(384).astype('float32') for _ in range(10)]
np_vectors = np.array(vectors).astype('float32')

index = faiss.IndexFlatL2(np_vectors.shape[1])  # Make sure dimension matches

URL='https://api.kroger.com/v1/connect/oauth2/token'
CREDS='Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI='
CT="application/x-www-form-urlencoded"
import  requests
token=dict(requests.post(url=URL, headers={'Content-Type': CT,"Authorization":CREDS},data={"grant_type":"client_credentials"}).json())["access_token"]
print(token)
@app.route("/stores")
def get_stores():
    zip_code = request.args.get("zip", default="45011")
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.zipCode.near": zip_code,
        "filter.radiusInMiles": "45",
        "filter.chain": "Kroger"
    }
    res = requests.get("https://api.kroger.com/v1/locations", headers=headers, params=params)
    return jsonify(res.json()), res.status_code

@app.route("/upload", methods=["POST"])
def upload_json():
    global product_metadata, index_is_empty,index
    
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided."}), 400

    all_descriptions = []
    new_meta = []

    for file in files:
        try:
            data = json.load(file)
            for item in data.get("data", []):
                desc = item.get("description", "").strip()
                if desc:
                    all_descriptions.append(desc)
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
            return jsonify({"error": str(e)}), 500

    if all_descriptions:
        vectors = embedding_model.encode(all_descriptions)
        index.add(np.array(vectors))
        index_is_empty = False
        product_metadata.extend(new_meta)

    return jsonify({
        "message": f"Indexed {len(new_meta)} new products.",
        "total": len(product_metadata)
    })

@app.route("/search2", methods=["POST"])
def vector_search():
    query = request.form.get("query")
    global index_is_empty
    if not query or index_is_empty:
        return jsonify({"error": "Query missing or no data in index"}), 400

    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)

    results = []
    for idx in I[0]:
        if idx < len(product_metadata):
            product = product_metadata[idx]
            results.append(product)

    return jsonify({"query": query, "results": results})

from flask import render_template

@app.route("/mealplan", methods=["POST"])
def generate_meal_plan():
    goal = request.form.get("goal")
    if not goal and not product_metadata:
        return jsonify({"error": "Goal or product data missing."}), 400

    # Use top 10 products for context
    top_products = "\n".join([f"- {p['description']} ({p['brand']}) [Buy here]({p['link']})" for p in product_metadata[:10]])

    prompt = f"""
You are a certified nutritionist creating a meal plan for a woman with Gestational Diabetes Mellitus (GDM) or borderline GDM.

The following products are available in her kitchen:
{top_products}

The user goal is:
"{goal}"

Create a 1- or 3-day GDM-friendly meal plan:
- Include 3 meals + 2 snacks per day
- Show ingredients, portion sizes , AND LINKS TO PRODUCT FOR BUY USING KROGER 
- Include Nutrition Facts per meal (calories, carbs, protein, fat)
- Explain how meals support blood sugar control
- End with total daily nutrition and practical tips

Format output in JSON.
"""

    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        meal_plan = json.loads(response.text.strip())
        return render_template("meal_plan.html", meal_plan=meal_plan)
    except Exception as e:
        return jsonify({"error": f"Gemini error: {str(e)}"}), 500

if __name__ == "__main__":
    print("🚀 GDM Vector + Gemini API Server running at http://127.0.0.1:5050")
    app.run(port=5050, debug=True,host="0.0.0.0")
