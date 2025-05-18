from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template_string
import json
import os

app = Flask(__name__)
os.makedirs("uploaded_jsons", exist_ok=True)

# === EMBEDDING SETUP (NO sentence_transformers) ===
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
    embeddings = sum_embeddings / sum_mask

    return embeddings.cpu().numpy()

# === FAISS SETUP ===
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === FLASK ROUTES ===
@app.route("/")
def index1():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>Semantic Product Search</title></head>
    <body>
        <h2>Upload JSON</h2>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" multiple required>
            <button type="submit">Upload</button>
        </form>




<h3>Find Store by ZIP → Then Search Products</h3>
<form method="post" action="/choose_store">
    <input type="text" name="zip" placeholder="ZIP code (e.g. 45011)" pattern="\d{5}" required>
    <input type="text" name="query" placeholder="Product (e.g. tofu)" required>
    <button type="submit">Find Stores & Search</button>
</form>




        <h3>Semantic Search</h3>
        <form method="post" action="/search">
            <input type="text" name="query" placeholder="e.g. healthy cereal" required>
            <button type="submit">Search</button>
        </form>
"""+home()+""""
    </body>
    </html>
    """)



URL='https://api.kroger.com/v1/connect/oauth2/token'
CREDS='Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI='
CT="application/x-www-form-urlencoded"
import  requests
token=dict(requests.post(url=URL, headers={'Content-Type': CT,"Authorization":CREDS},data={"grant_type":"client_credentials","scope":"product.compact"}).json())["access_token"]



#@app.route("/search_by_store", methods=["POST"])
@app.route("/search_by_store", methods=["POST"])
def search_by_store():
    query = request.form.get("query")
    location_id = request.form.get("locationId")

    if not query or not location_id:
        return "Missing query or locationId", 400

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.term": query,
        "filter.locationId": location_id,
        "filter.fulfillment": "dth",
        "filter.limit": "10"
    }

    response = requests.get("https://api.kroger.com/v1/products", headers=headers, params=params)
    if response.status_code != 200:
        return f"Kroger API error: {response.text}", 500

    products = response.json().get("data", [])
    if not products:
        return f"No '{query}' found at store {location_id}."

    def parse_oz_or_lb(size_str):
        """Extract ounces or pounds from size string"""
        import re
        match = re.search(r"([\d.]+)\s*(oz|lb)", size_str.lower())
        if match:
            qty = float(match.group(1))
            unit = match.group(2)
            if unit == "oz":
                return qty / 16  # convert oz to lb
            return qty  # already in lb
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
            if weight_lb and weight_lb > 0:
                price_per_lb = price / weight_lb
                per_lb = f" (${price_per_lb:.2f}/lb)"

        html += "<li>"
        if image:
            html += f'<img src="{image}" style="height:40px; vertical-align:middle;"> '
        html += f"<strong>{desc}</strong> – {brand} – {size}"
        if price:
            html += f" – ${price:.2f}{per_lb}"
        html += "</li>"
    html += "</ul><p><a href='/'>Search again</a></p>"
    return html


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

# Home route with form to input goal
#@app.route("/", methods=["GET"])
def home():
    return """
        <h1>GDM Meal Planner</h1>
        <form method="post" action="/mealplan">
            <input type="text" name="goal" placeholder="Enter your meal plan goal" required>
            <button type="submit">Generate Meal Plan</button>
        </form>
    """

if __name__ == "__main__":
    app.run(debug=True)


#@app.route("/search_by_store", methods=["POST"])
def search_by_store2():
    query = request.form.get("query")
    location_id = request.form.get("locationId")

    if not query or not location_id:
        return "Missing query or locationId", 400

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.term": query,
        "filter.locationId": location_id,
        "filter.fulfillment": "dth",
        "filter.limit": "10"
    }

    response = requests.get("https://api.kroger.com/v1/products", headers=headers, params=params)
    if response.status_code != 200:
        return f"Kroger API error: {response.text}", 500

    products = response.json().get("data", [])
    if not products:
        return f"No '{query}' found at store {location_id}."

    def parse_oz_or_lb(size_str):
        """Extract ounces or pounds from size string"""
        import re
        match = re.search(r"([\d.]+)\s*(oz|lb)", size_str.lower())
        if match:
            qty = float(match.group(1))
            unit = match.group(2)
            if unit == "oz":
                return qty / 16  # convert oz to lb
            return qty  # already in lb
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
            if weight_lb and weight_lb > 0:
                price_per_lb = price / weight_lb
                per_lb = f" (${price_per_lb:.2f}/lb)"

        html += "<li>"
        if image:
            html += f'<img src="{image}" style="height:40px; vertical-align:middle;"> '
        html += f"<strong>{desc}</strong> – {brand} – {size}"
        if price:
            html += f" – ${price:.2f}{per_lb}"
        html += "</li>"
    html += "</ul><p><a href='/'>Search again</a></p>"
    return html




def searcha_by_store2():
    query = request.form.get("query")
    location_id = request.form.get("locationId")

    if not query or not location_id:
        return "Missing query or locationId", 400

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.term": query,
        "filter.locationId": location_id,
        "filter.fulfillment": "dth",
        "filter.limit": "10"
    }

    response = requests.get("https://api.kroger.com/v1/products", headers=headers, params=params)
    if response.status_code != 200:
        return f"Kroger API error: {response.text}", 500

    products = response.json().get("data", [])
    if not products:
        return f"No '{query}' found at store {location_id}."

    # Sort by product description (optional)
    products = sorted(products, key=lambda p: p.get("description", ""))

    html = f"<h3>Results for '{query}' at Store ID {location_id}</h3><ul>"
    for p in products:
        desc = p.get("description", "Unknown")
        brand = p.get("brand", "")
        price = p.get("items", [{}])[0].get("price", {}).get("regular")
        image = p.get("images", [{}])[0].get("sizes", [{}])[0].get("url", "")
        html += "<li>"
        if image:
            html += f'<img src="{image}" alt="{desc}" style="height:40px;"> '
        html += f"<strong>{desc}</strong> – {brand}"
        if price:
            html += f" – ${price:.2f}"
        html += "</li>"
    html += "</ul><p><a href='/'>Search again</a></p>"
    return html



#@app.route("/search_1by_store", methods=["POST"])
def s8earch_by_store():
    query = request.form.get("query")
    location_id = request.form.get("locationId")

    if not query or not location_id:
        return "Missing query or locationId", 400

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.term": query,
        "filter.locationId": location_id,
        "filter.fulfillment": "dth",
        "filter.limit": "10"
    }

    response = requests.get("https://api.kroger.com/v1/products", headers=headers, params=params)
    if response.status_code != 200:
        return f


@app.route("/choose_store", methods=["POST"])
def choose_store():
    zip_code = request.form.get("zip")
    query = request.form.get("query")
    if not zip_code or not query:
        return "ZIP and query required", 400

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.zipCode.near": zip_code,
        "filter.radiusInMiles": "20",
        "filter.limit": 10,
        "filter.chain": "Kroger"
    }
    res = requests.get("https://api.kroger.com/v1/locations", headers=headers, params=params)
    stores = res.json().get("data", [])

    if not stores:
        return "No stores found for this ZIP.", 404

    # Show list of radio buttons
    html = "<h3>Select a store near ZIP {}</h3>".format(zip_code)
    html += '<form method="post" action="/search_by_store">'
    html += f'<input type="hidden" name="query" value="{query}">'
    for store in stores:
        lid = store["locationId"]
        address = store["address"].get("addressLine1", "")
        city = store["address"].get("city", "")
        state = store["address"].get("state", "")
        html += f'<label><input type="radio" name="locationId" value="{lid}" required> {lid} - {address}, {city}, {state}</label><br>'
    html += '<br><button type="submit">Search in Selected Store</button></form>'
    return html


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
    global product_metadata, index_is_empty, index

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
                        "category": ", ".join(item.get("categories", [])) if item.get("categories") else "",
                        "image": next((img["sizes"][0]["url"]
                            for img in item.get("images", []) if img.get("sizes")), ""),
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
    global index_is_empty
    query = request.form.get("query")
    if not query or index_is_empty:
        return jsonify({"error": "Query missing or no data in index"}), 400

    query_vec = encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)

    results = []
    for idx in I[0]:
        if idx < len(product_metadata):
            product = product_metadata[idx]
            results.append(product)

    return jsonify({"query": query, "results": results})

if __name__ == "__main__":
    print("🚀 Running Flask app at http://127.0.0.1:4401")
    app.run(host="0.0.0.0",port=4401)


