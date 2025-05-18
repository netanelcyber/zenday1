import os
import json
import requests
from time import sleep

# === Constants ===
KROGER_CREDS = "Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI="
TOKEN_URL = "https://api.kroger.com/v1/connect/oauth2/token"
PRODUCTS_URL = "https://api.kroger.com/v1/products"
SAVE_DIR = "kroger_products_nutrition"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_kroger_token():
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": KROGER_CREDS}
    data = {"grant_type": "client_credentials", "scope": "product.compact"}
    res = requests.post(TOKEN_URL, headers=headers, data=data)
    res.raise_for_status()
    return res.json()["access_token"]

def fetch_products_with_nutrition(token, keyword, max_products=10000):
    headers = {"Authorization": f"Bearer {token}"}
    all_products = []
    start = 0
    page = 0
    batch_size = 50  # max per page
    total_fetched = 0

    while total_fetched < max_products:
        params = {
            "filter.term": keyword,
            "filter.limit": batch_size,
            "filter.start": start
        }

        response = requests.get(PRODUCTS_URL, headers=headers, params=params)
        if response.status_code != 200:
            print(f"[ERROR] {response.status_code} on page {page}: {response.text}")
            break

        data = response.json().get("data", [])
        if not data:
            print("[INFO] No more products returned.")
            break

        # Only keep products with nutrition facts
        for product in data:
            if product.get("items"):
                for item in product["items"]:
                    nutrition = item.get("nutrition", {}).get("nutritionFacts")
                    if nutrition:
                        product["nutritionFacts"] = nutrition
                        all_products.append(product)
                        break  # only need one valid nutrition-containing item per product

        # Save every 500 products to file
        if len(all_products) % 500 < batch_size:
            save_path = os.path.join(SAVE_DIR, f"{keyword.replace(' ', '_')}_batch_{page}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_products[-500:], f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved batch {page} with {len(all_products)} total")

        total_fetched += len(data)
        start += batch_size
        page += 1
        sleep(0.5)  # to prevent throttling

    print(f"[DONE] Fetched {len(all_products)} products with nutrition for '{keyword}'.")

# === Run for one or many keywords ===
if __name__ == "__main__":
    token = get_kroger_token()
    search_keywords = ["healthy", "low sugar", "organic", "diabetes", "protein", "vegetarian"]
    for kw in search_keywords:
        fetch_products_with_nutrition(token, kw, max_products=10000)
