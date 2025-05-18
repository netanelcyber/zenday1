import os
import json
import requests
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from bottle import route, run, request, response, static_file

# === CONFIG ===
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your actual Gemini API key
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === Kroger Auth ===
URL = 'https://api.kroger.com/v1/connect/oauth2/token'
CREDS = "Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI="

CT = "application/x-www-form-urlencoded"
token = requests.post(url=URL, headers={'Content-Type': CT, "Authorization": CREDS},
                      data={"grant_type": "client_credentials","scope":"product.compact"}).json().get("access_token", "")
print(token)
# === ROUTES ===

@route('/')
def index():
    return """
    <h2>Upload Kroger JSON</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" multiple required><button type="submit">Upload</button>
    </form>
    <h3>Search Product by ZIP</h3>
    <form action="/search" method="post">
        <input name="query" required placeholder="e.g. almond milk">
        <input name="zip" required pattern="\\d{5}" placeholder="ZIP Code">
        <button>Search</button>
    </form>
    <h3>Semantic Search</h3>
    <form action="/search2" method="post">
        <input name="query" required placeholder="e.g. healthy snacks"><button>Search</button>
    </form>
    <h3>Generate GDM Meal Plan</h3>
    <form action="/mealplan" method="post">
        <input name="goal" required placeholder="e.g. 3-day meal plan with snacks"><button>Generate</button>
    </form>
    """

@route('/upload', method='POST')
def upload_json():
    global product_metadata, index_is_empty, index
    files = request.files.getall('file')
    if not files:
        response.status = 400
        return {"error": "No files provided"}

    all_descriptions = []
    new_meta = []

    for file in files:
        raw = file.file.read().decode('utf-8')
        try:
            data = json.loads(raw)
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
            response.status = 500
            return {"error": str(e)}

    if all_descriptions:
        vectors = embedding_model.encode(all_descriptions)
        index.add(np.array(vectors))
        index_is_empty = False
        product_metadata.extend(new_meta)

    return {"message": f"Indexed {len(new_meta)} products", "total": len(product_metadata)}

@route('/search', method='POST')
def search_zip():
    q = request.forms.get("query")
    zip_code = request.forms.get("zip")

    if not q or not zip_code:
        response.status = 400
        return {"error": "Query and ZIP code required"}

    headers = {"Authorization": f"Bearer {token}"}
    store_res = requests.get("https://api.kroger.com/v1/locations", headers=headers, params={
        "filter.zipCode.near": zip_code,
        "filter.radiusInMiles": "35",
        "filter.chain": "Kroger"
    })

    stores = store_res.json().get("data", [])
    return json.dumps(stores)
    if not stores:
        return {"error": "No stores found near ZIP"}

    results = []
    for store in stores:
        location_id = store.get("locationId")
        print(location_id)
        prod_res = requests.get(f"https://api.kroger.com/v1/products/?filter.term={q}&filter.location={location_id}&filter.limit=50", headers=headers)

        print(prod_res.json())
        results.append(prod_res.json())
        for product in prod_res.json().get("data", []):
            results.append({
                "store": store.get("address", {}).get("addressLine1", "Unknown"),
                "product": {
                    "description": product.get("description"),
                    "brand": product.get("brand"),
                    "price": product.get("items", [{}])[0].get("price", {}).get("regular"),
                    "image": product.get("images", [{}])[0].get("sizes", [{}])[0].get("url"),
                    "link": f"https://www.kroger.com/p/{product.get('description', '').replace(' ', '-').lower()}/{product.get('productId')}"
                }
            })

    return {"query": q, "results": prod_res}

@route('/search2', method='POST')
def vector_search():
    q = request.forms.get("query")
    global index_is_empty
    if not q or index_is_empty:
        response.status = 400
        return {"error": "Query missing or no index"}

    query_vec = embedding_model.encode([q])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)

    results = []
    for idx in I[0]:
        if idx < len(product_metadata):
            results.append(product_metadata[idx])

    return {"query": q, "results": results}

@route('/mealplan', method='POST')
def generate_meal_plan():
    goal = request.forms.get("goal")
    if not goal:
        response.status = 400
        return {"error": "Missing goal"}

    top_products = "\n".join([
        f"- {p['description']} ({p['brand']}) [Buy here]({p['link']})"
        for p in product_metadata[:10]
    ])

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
        model = genai.GenerativeModel("gemini-1.5-pro")
        result = model.generate_content(prompt)
        meal_plan = json.loads(result.text.strip())
        response.content_type = 'application/json'
        return json.dumps(meal_plan, indent=2)
    except Exception as e:
        response.status = 500
        return {"error": f"Gemini error: {str(e)}"}

# === Run Bottle app ===
if __name__ == "__main__":
    print("🚀 GDM Bottle server running")
    run(host='0.0.0.0', port=9977, debug=True, reloader=True)
