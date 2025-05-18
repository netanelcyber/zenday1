import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext
import json
import requests
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# === Setup ===
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace this with your real key
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === Kroger Token Setup ===
URL = 'https://api.kroger.com/v1/connect/oauth2/token'
CREDS = "Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI="
CT = "application/x-www-form-urlencoded"

try:
    token_res = requests.post(
        url=URL,
        headers={'Content-Type': CT, "Authorization": CREDS},
        data={"grant_type": "client_credentials", "scope": "product.compact"}
    )
    kroger_token = token_res.json().get("access_token", "")
    if kroger_token:
        print("✅ Kroger token acquired.")
    else:
        raise ValueError("Token missing in response")
except Exception as e:
    print("❌ Kroger token error:", str(e))
    kroger_token = ""

# === GUI ===
root = tk.Tk()
root.title("GDM Meal Planner")
root.geometry("1000x700")

output = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=35)
output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

def log(msg):
    output.insert(tk.END, f"{msg}\n")
    output.see(tk.END)

# === Core Features ===

def upload_json():
    global index_is_empty
    filepaths = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])
    if not filepaths:
        return

    all_descriptions = []
    new_meta = []

    for path in filepaths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
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
            messagebox.showerror("Error", f"Failed to load JSON: {str(e)}")
            return

    if all_descriptions:
        vectors = embedding_model.encode(all_descriptions)
        index.add(np.array(vectors).astype("float32"))
        product_metadata.extend(new_meta)
        index_is_empty = False
        log(f"✅ Indexed {len(new_meta)} products.")
    else:
        log("⚠️ No valid product descriptions found.")

def semantic_search():
    if index_is_empty:
        messagebox.showwarning("No Data", "Please upload data first.")
        return
    query = simpledialog.askstring("Semantic Search", "Enter search query:")
    if not query:
        return

    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)
    results = [product_metadata[i] for i in I[0] if i < len(product_metadata)]

    log(f"\n🔍 Top matches for '{query}':\n")
    for r in results:
        log(f"- {r['description']} ({r['brand']})\n  Link: {r['link']}\n")

def generate_meal_plan():
    if not product_metadata:
        messagebox.showwarning("No Data", "Please upload product data first.")
        return
    goal = simpledialog.askstring("Meal Goal", "Enter GDM-friendly meal goal:")
    if not goal:
        return

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
- Show ingredients, portion sizes, AND LINKS TO PRODUCT FOR BUY USING KROGER
- Include Nutrition Facts per meal (calories, carbs, protein, fat)
- Explain how meals support blood sugar control
- End with total daily nutrition and practical tips

Format output in JSON.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        result = model.generate_content(prompt)
        meal_plan = json.loads(result.text.strip())
        log("\n🍽️ Generated Meal Plan:\n")
        log(json.dumps(meal_plan, indent=2))
    except Exception as e:
        messagebox.showerror("Gemini Error", str(e))

def search_kroger_by_zip():
    if not kroger_token:
        messagebox.showerror("Token Error", "Kroger token is missing.")
        return

    zip_code = simpledialog.askstring("ZIP Code", "Enter ZIP code:")
    query = simpledialog.askstring("Search", "Enter product name:")
    if not zip_code or not query:
        return

    try:
        headers = {"Authorization": f"Bearer {kroger_token}"}
        store_res = requests.get("https://api.kroger.com/v1/locations", headers=headers, params={
            "filter.zipCode.near": zip_code,
            "filter.radiusInMiles": "35",
            "filter.chain": "Kroger"
        })
        stores = store_res.json().get("data", [])
        if not stores:
            log("❌ No Kroger stores found near that ZIP.")
            return

        for store in stores[:1]:
            loc_id = store.get("locationId")
            store_name = store.get("address", {}).get("addressLine1", "Unknown")
            product_res = requests.get("https://api.kroger.com/v1/products", headers=headers, params={
                "filter.term": query,
                "filter.locationId": loc_id,
                "filter.limit": 10
            })
            log(f"\n🛒 {store_name} | Top '{query}' results:\n")
            for p in product_res.json().get("data", []):
                price = p.get("items", [{}])[0].get("price", {}).get("regular")
                link = f"https://www.kroger.com/p/{p.get('description', '').replace(' ', '-').lower()}/{p.get('productId')}"
                log(f"- {p['description']} ({p['brand']})\n  ${price} | {link}")
    except Exception as e:
        messagebox.showerror("Kroger API Error", str(e))

# === Buttons ===
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="📂 Upload Kroger JSON", command=upload_json).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="🔍 Semantic Search", command=semantic_search).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="🍽️ Generate Meal Plan", command=generate_meal_plan).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="🛒 Kroger ZIP Search", command=search_kroger_by_zip).pack(side=tk.LEFT, padx=10)

# === Run App ===
log("👋 Welcome to the GDM Meal Planner (Tkinter Edition)")
root.mainloop()
