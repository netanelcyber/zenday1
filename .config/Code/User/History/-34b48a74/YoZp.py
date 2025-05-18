from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import requests
from flask import Flask, request, jsonify, render_template_string
import json, os, re

app = Flask(__name__)
os.makedirs("uploaded_jsons", exist_ok=True)
api_key="AIzaSyCq7zNTN3pebYJB2f-BT2t031pvwc85t8c"
# === Embedding Setup (No sentence_transformers) ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    token_embeddings = model_output.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().numpy()

# === FAISS Vector Index ===
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === Kroger API Token Setup ===
KROGER_CREDS = "Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI="
token_url = "https://api.kroger.com/v1/connect/oauth2/token"
token_headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": KROGER_CREDS}
token_data = {"grant_type": "client_credentials", "scope": "product.compact"}
token = requests.post(token_url, headers=token_headers, data=token_data).json()["access_token"]
KROGER_HEADERS = {"Authorization": f"Bearer {token}"}

# === Routes ===


@app.route("/mealplan", methods=["POST"])
def generate_meal_plan():
    goal = request.form.get("goal")
    if not goal:
        return "Goal is required.", 400

    # Call Gemini API to generate meal plan
    # Replace this with actual API call
    # For demonstration, using a mock response
    plan_text = f"Meal plan for: {goal}"

    # Extract product links
    links = re.findall(r"https://www\.kroger\.com/p/[^)\s]+", plan_text)

    return render_template_string("""
        <h2>GDM Meal Plan</h2>
        <pre>{{ plan }}</pre>
        {% if links %}
        <h3>Select Products to Buy</h3>
        <form method="post" action="/optimize_buy">
            {% for link in links %}
              <label><input type="checkbox" name="product_links" value="{{ link }}"> {{ link }}</label><br>
            {% endfor %}
            <button type="submit">Find Cheapest Options</button>
        </form>
        {% endif %}
    """, plan=plan_text, links=links)

# Route to optimize and display products
@app.route("/optimize_buy", methods=["POST"])
def optimize_buy():
    links = request.form.getlist("product_links")
    if not links:
        return "No products selected.", 400

    products = []
    for link in links:
        try:
            slug = link.split("/p/")[1]
            desc = slug.split("/")[0].replace("-", " ")
        except:
            continue

        params = {"filter.term": desc, "filter.limit": "1"}
        res = requests.get("https://api.kroger.com/v1/products", headers=KROGER_HEADERS, params=params)
        if res.status_code != 200:
            continue
        data = res.json().get("data", [])
        if not data:
            continue
        product = data[0]
        upc = product.get("items", [{}])[0].get("upc")
        name = product.get("description")
        price = product.get("items", [{}])[0].get("price", {}).get("regular")
        products.append({"name": name, "upc": upc, "price": price})

    return render_template_string("""
        <h2>Optimized Products to Add to Cart</h2>
        <form method="post" action="/add_to_cart">
            {% for p in products %}
              <label>
                <input type="checkbox" name="upcs" value="{{ p.upc }}"> 
                {{ p.name }} – ${{ "%.2f"|format(p.price or 0) }}
              </label><br>
            {% endfor %}
            <br><button type="submit">Add to Cart</button>
        </form>
    """, products=products)

# Route to add products to cart
@app.route("/add_to_cart", methods=["POST"])
def add_to_cart():
    upcs = request.form.getlist("upcs")
    if not upcs:
        return "No UPCs selected.", 400

    payload = {
        "items": [{"quantity": 1, "upc": upc, "modality": "DELIVERY"} for upc in upcs]
    }

    response = requests.put("https://api.kroger.com/v1/cart/add", headers=KROGER_HEADERS, json=payload)
    if response.status_code in [200, 201, 204]:
        return "<h2>Items successfully added to your Kroger cart!</h2><p><a href='/'>Back to Home</a></p>"
    return f"Failed to add to cart: {response.status_code} - {response.text}"


@app.route("/")
def index():
    return render_template_string(f"""
        <!DOCTYPE html>
        <html><head><title>Semantic Product Search</title></head>
        <body>
            <h2>Upload JSON</h2>
            <form method="post" action="/upload" enctype="multipart/form-data">
                <input type="file" name="file" multiple required>
                <button type="submit">Upload</button>
            </form>
            <h3>Find Store by ZIP → Then Search Products</h3>
            <form method="post" action="/choose_store">
                <input type="text" name="zip" placeholder="ZIP code" pattern="\\d{{5}}" required>
                <input type="text" name="query" placeholder="Product (e.g. tofu)" required>
                <button type="submit">Find Stores & Search</button>
            </form>
            <h3>Semantic Search</h3>
            <form method="post" action="/search">
                <input type="text" name="query" placeholder="e.g. healthy cereal" required>
                <button type="submit">Search</button>
            </form>
            {home()}
        </body></html>
    """)

@app.route("/home")
def home():
    return """
        <h1>GDM Meal Planner</h1>
        <form method="post" action="/mealplan">
            <input type="text" name="goal" placeholder="Enter your meal plan goal" required>
            <button type="submit">Generate Meal Plan</button>
        </form>
    """

@app.route("/upload", methods=["POST"])
def upload_json():
    global product_metadata, index_is_empty

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided."}), 400

    all_descriptions, new_meta = [], []
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
                        "category": ", ".join(item.get("categories", [])) if item.get("categories") else "",
                        "image": next((img["sizes"][0]["url"] for img in item.get("images", []) if img.get("sizes")), ""),
                        "link": f"https://www.kroger.com/p/{desc.replace(' ', '-').lower()}/{item.get('productId')}"
                    })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if all_descriptions:
        vectors = encode(all_descriptions)
        index.add(np.array(vectors).astype("float32"))
        index_is_empty = False
        product_metadata.extend(new_meta)

    return jsonify({
        "message": f"Indexed {len(new_meta)} new products.",
        "total": len(product_metadata)
    })

@app.route("/search", methods=["POST"])
def vector_search():
    if index_is_empty:
        return jsonify({"error": "No products in index."}), 400

    query = request.form.get("query")
    if not query:
        return jsonify({"error": "Query is required."}), 400

    query_vec = encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)
    results = [product_metadata[i] for i in I[0] if i < len(product_metadata)]
    return jsonify({"query": query, "results": results})

@app.route("/choose_store", methods=["POST"])
def choose_store():
    zip_code, query = request.form.get("zip"), request.form.get("query")
    if not zip_code or not query:
        return "ZIP and query required", 400

    params = {
        "filter.zipCode.near": zip_code,
        "filter.radiusInMiles": "20",
        "filter.limit": 10,
        "filter.chain": "Kroger"
    }
    res = requests.get("https://api.kroger.com/v1/locations", headers=KROGER_HEADERS, params=params)
    stores = res.json().get("data", [])

    if not stores:
        return "No stores found for this ZIP.", 404

    html = f"<h3>Select a store near ZIP {zip_code}</h3>"
    html += '<form method="post" action="/search_by_store">'
    html += f'<input type="hidden" name="query" value="{query}">'
    for store in stores:
        lid = store["locationId"]
        addr = store["address"]
        html += f'<label><input type="radio" name="locationId" value="{lid}" required> {lid} - {addr.get("addressLine1", "")}, {addr.get("city", "")}, {addr.get("state", "")}</label><br>'
    html += '<br><button type="submit">Search in Selected Store</button></form>'
    return html

@app.route("/search_by_store", methods=["POST"])
def search_by_store():
    query = request.form.get("query")
    location_id = request.form.get("locationId")
    if not query or not location_id:
        return "Missing query or locationId", 400

    params = {
        "filter.term": query,
        "filter.locationId": location_id,
        "filter.fulfillment": "dth",
        "filter.limit": "10"
    }
    response = requests.get("https://api.kroger.com/v1/products", headers=KROGER_HEADERS, params=params)
    if response.status_code != 200:
        return f"Kroger API error: {response.text}", 500

    products = response.json().get("data", [])
    if not products:
        return f"No '{query}' found at store {location_id}."

    def parse_oz_or_lb(size_str):
        match = re.search(r"([\d.]+)\s*(oz|lb)", size_str.lower())
        if match:
            qty, unit = float(match.group(1)), match.group(2)
            return qty / 16 if unit == "oz" else qty
        return None

    html = f"<h3>Results for '{query}' at Store ID {location_id}</h3><ul>"
    for p in sorted(products, key=lambda x: x.get("description", "")):
        desc = p.get("description", "Unknown")
        brand = p.get("brand", "")
        items = p.get("items", [])
        image = p.get("images", [{}])[0].get("sizes", [{}])[0].get("url", "")
        size = p.get("size", "") or (items[0].get("size") if items else "")
        price = items[0].get("price", {}).get("regular") if items else None
        per_lb = ""
        if price and size:
            weight_lb = parse_oz_or_lb(size)
            if weight_lb:
                per_lb = f" (${price / weight_lb:.2f}/lb)"
#        html += f"<li>{f'<img src={image} style=\"height:40px;\"> ' if image else ''}<strong>"+desc+"</strong> – "+brand+" – "+size+f"{f' – ${price:.2f}{per_lb}' if price else ''}</li>"
#        html += f"<li>{f'<img src=\"{image}\" style=\"height:40px;\"> ' if image else ''}<strong>{desc}</strong> – {brand} – {size}{f' – ${price:.2f}{per_lb}' if price else ''}</li>"
        html += "<li>"
        if image:
           html += f'<img src="{image}" style="height:40px;"> '
        html += f"<strong>{desc}</strong> – {brand} – {size}"
        if price:
           html += f" – ${price:.2f}{per_lb}"
        html += "</li>"

        html += "</ul><p><a href='/'>Search again</a></p>"
    return html



@app.route("/model")
def generate_meal_plan_p():
    goal = request.form.get("goal")
    if not goal:
        goal="reduce sugar levels,vegan" 

    # === Step 1: Generate Meal Plan from HF LSTM ===
    prompt = f"""
Create a 3-day GDM-friendly meal plan based on the goal: "{goal}".
Each day should have 3 meals and 2 snacks.
Each item should include name and quantity.
Return result in JSON format:
{{ "days": [{{"day": 1, "meals": [{{"name": "Breakfast", "items": [{{"name": "Oatmeal", "amount": "1 cup"}}]}}]}}] }}
"""

    hf_response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
        headers=HF_HEADERS,
        json={"inputs": prompt}
    )

    if hf_response.status_code != 200:
        return jsonify({"error": "HF API failed", "details": hf_response.text}), 500

    try:
        raw = hf_response.json()
        text = raw[0]["generated_text"] if isinstance(raw, list) else raw.get("generated_text", "")
        plan = json.loads(text)
    except Exception as e:
        return jsonify({"error": "Parsing error", "raw": text, "details": str(e)}), 500

    # === Step 2: Kroger Token ===
    #token_data = {"grant_type": "client_credentials", "scope": "product.compact"}
    #token_headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": KROGER_CREDS}
    #token_res = requests.post(TOKEN_URL, headers=token_headers, data=token_data)
    #kroger_token = token_res.json()["access_token"]
    #KROGER_HEADERS = {"Authorization": f"Bearer {kroger_token}"}

    # === Step 3: Search Kroger Products ===
    ingredient_set = set()
    for day in plan.get("days", []):
        for meal in day.get("meals", []):
            for item in meal.get("items", []):
                ingredient_set.add(item["name"])

    enriched = []
    for name in ingredient_set:
        params = {"filter.term": name, "filter.limit": 1}
        res = requests.get("https://api.kroger.com/v1/products", headers=KROGER_HEADERS, params=params)
        product = res.json().get("data", [None])[0]
        if not product:
            continue

        image = product.get("images", [{}])[0].get("sizes", [{}])[0].get("url", "")
        items = product.get("items", [])
        nutrition = items[0].get("nutrition", {}).get("nutritionFacts", []) if items else []

        enriched.append({
            "name": product.get("description", name),
            "brand": product.get("brand", ""),
            "productId": product.get("productId", ""),
            "link": f"https://www.kroger.com/p/{quote_plus(product.get('description', name).lower().replace(' ', '-'))}/{product.get('productId')}",
            "image": image,
            "nutrition": nutrition
        })

    # === Step 4: Render Output ===
    html = "<h2>Enriched Meal Plan Products</h2><ul>"
    for item in enriched:
        html += f"<li><img src='{item['image']}' width='40'> <strong>{item['name']}</strong> by {item['brand']}"
        html += f" – <a href='{item['link']}' target='_blank'>View Product</a>"
        if item['nutrition']:
            html += "<ul>"
            for n in item['nutrition']:
                html += f"<li>{n.get('description', '')}: {n.get('value', '')} {n.get('unit', '')}</li>"
            html += "</ul>"
        html += "</li>"
    html += "</ul><p><a href='/'>Back</a></p>"

    return html



@app.route("/mealplan", methods=["POST"])
def generate_meal_plan1():
    import google.generativeai as genai
    from urllib.parse import quote_plus

# === Gemini Setup ===
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    goal = request.form.get("goal")
    if not goal:
        return "Missing goal", 400

    prompt = f"""
You are a certified nutritionist creating a GDM meal plan.

The user goal is: "{goal}"

Please return a 3-day meal plan (3 meals + 2 snacks/day) with:
- Ingredients (with name and quantity)
- Nutrition facts
- Short description of why it helps blood sugar
- Link each ingredient to a real Kroger product if possible (use search terms)

Respond in JSON format like:
{{
  "days": [{{"day": 1, "meals": [{{"name": "...", "items": [{{"name": "...", "amount": "1 cup"}}]}}]}}]
}}
"""

    try:
        gemini_response = gemini_model.generate_content(prompt)
        print(prompt)
        print(gemini_response)
        plan = json.loads(gemini_response.text)
    except Exception as e:
        return jsonify({"error": "Gemini error", "details": str(e)}), 500

    # Extract ingredient names
    ingredients = []
    for day in plan.get("days", []):
        for meal in day.get("meals", []):
            for item in meal.get("items", []):
                name = item.get("name")
                if name and name not in ingredients:
                    ingredients.append(name)

    # Search for each ingredient on Kroger
    token = get_kroger_token()
    kroger_results = []
    for ing in ingredients:
        product = search_kroger(ing, token)
        if product:
            desc = product.get("description", ing)
            link = f"https://www.kroger.com/p/{quote_plus(desc.lower().replace(' ', '-'))}/{product.get('productId')}"
            kroger_results.append({
                "name": ing,
                "product": desc,
                "brand": product.get("brand"),
                "link": link,
                "image": product.get("images", [{}])[0].get("sizes", [{}])[0].get("url", "")
            })

    # Display nicely
    html = "<h2>Recommended Products from Meal Plan</h2><ul>"
    for p in kroger_results:
        html += f"<li><img src='{p['image']}' width='40'> <strong>{p['product']}</strong> by {p['brand']} — <a href='{p['link']}' target='_blank'>View</a></li>"
    html += "</ul><p><a href='/'>Back</a></p>"

    return html



if __name__ == "__main__":
    print("🚀 Running Flask app at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0")
