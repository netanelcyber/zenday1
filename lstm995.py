# LSTM meal planner using Open Food Facts API with nutrition-based logistic scoring



import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from collections import Counter

import re

import json

import os

import math

import time

import requests

from requests.adapters import HTTPAdapter

from urllib3.util.retry import Retry

from bs4 import BeautifulSoup

import argparse

import random



# === Setup requests with retry mechanism ===

def get_session():

    session = requests.Session()

    retries = Retry(

        total=5,

        backoff_factor=1,

        status_forcelist=[429, 500, 502, 503, 504],

        allowed_methods=["GET"]

    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    return session



# User agent rotation to avoid being blocked

USER_AGENTS = [

    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',

    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',

    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',

    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0',

    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'

]



# === Load Open Food Facts ===

def logistic(x, midpoint, steepness):

    return 1 / (1 + math.exp(-steepness * (x - midpoint)))



def score_product(nutri):

    score = 0

    weights = {

        'fiber_100g': (5, 1.0, 1.5),            # encourage high fiber

        'carbohydrates_100g': (45, -1.0, 0.1),  # penalize excess carbs (pref. complex)

        'sugars_100g': (5, -1.2, 0.4),           # minimize simple sugars

        'proteins_100g': (20, 1.0, 0.2),         # support moderate/high protein

        'fat_100g': (30, -0.5, 0.1),             # prefer unsaturated, avoid excess

        'saturated_fat_100g': (10, -1.0, 0.3)    # penalize saturated fat

    }

    for key, (mid, weight, k) in weights.items():

        if key in nutri:

            score += weight * logistic(nutri[key], mid, k)

    return score / sum(abs(w[1]) for w in weights.values())



def load_openfoodfacts_jsonl(path, min_grade=0.6, max_items=15000):

    descriptions = []

    with open(path, 'r', encoding='utf-8') as f:

        for line in f:

            try:

                product = json.loads(line)

                name = product.get("product_name", "").lower()

                nutri = product.get("nutriments", {})

                if not name or not nutri:

                    continue



                grade = score_product(nutri)

                if grade >= min_grade:

                    facts = [f"{k}:{nutri[k]}" for k in ["fiber_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g"] if k in nutri]

                    desc = f"{name} | {' '.join(facts)}"

                    descriptions.append(desc)



                if len(descriptions) >= max_items:

                    break

            except Exception:

                continue

    with open("scraped_products.jsonl", "w", encoding="utf-8") as fout:

        for line in descriptions:

            fout.write(json.dumps({"description": line}) + "\n")

    return descriptions



# === Tokenizer & Vocab ===

def tokenize(text):

    return re.findall(r"\b\w+\b", text.lower())



def build_vocab(descriptions, min_freq=1):

    counter = Counter()

    for text in descriptions:

        counter.update(tokenize(text))

    vocab = {word: i + 2 for i, (word, count) in enumerate(counter.items()) if count >= min_freq}

    vocab["<pad>"] = 0

    vocab["<eos>"] = 1

    return vocab



# === Dataset ===

class ProductDataset(Dataset):

    def __init__(self, texts, vocab, max_len=15):

        self.vocab = vocab

        self.max_len = max_len

        self.data = [self.encode(t) for t in texts if t.strip()]



    def encode(self, text):

        tokens = tokenize(text)[:self.max_len - 1]

        ids = [self.vocab.get(t, 0) for t in tokens]

        ids.append(self.vocab["<eos>"])

        ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))

        return torch.tensor(ids)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        x = self.data[idx][:-1]

        y = self.data[idx][1:]

        return x, y



# === Model ===

class LSTMTextGen(nn.Module):

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)



    def forward(self, x, hidden=None):

        x = self.embed(x)

        out, hidden = self.lstm(x, hidden)

        out = self.fc(out)

        return out, hidden



# === Train Function ===

def train(model, dataloader, vocab_size, epochs=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)



    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out, _ = model(x)

            loss = loss_fn(out.view(-1, vocab_size), y.view(-1))

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")



# === Generate Text ===

def generate(model, vocab, prompt, max_len=20):

    idx2word = {i: w for w, i in vocab.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()



    tokens = tokenize(prompt)

    ids = [vocab.get(t, 0) for t in tokens]

    if not ids:

        return "[Invalid prompt]"

    input_tensor = torch.tensor([ids], dtype=torch.long).to(device)

    output = tokens[:]

    hidden = None



    with torch.no_grad():

        for _ in range(max_len):

            out, hidden = model(input_tensor, hidden)

            next_id = out[0, -1].argmax().item()

            if next_id == vocab["<eos>"]:

                break

            word = idx2word.get(next_id, "?")

            output.append(word)

            input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(device)



    return " ".join(output)



# === Use Open Food Facts API instead of web scraping ===

def fetch_from_openfoodfacts_api(queries, max_products_per_query=50):

    descriptions = []

    session = get_session()

    

    # API base URL

    api_url = "https://world.openfoodfacts.org/api/v0/product/"

    

    # Also search by category/tag

    search_url = "https://world.openfoodfacts.org/cgi/search.pl"

    

    for query in queries:

        print(f"Searching Open Food Facts API for: {query}")

        try:

            # Random user agent for each request

            headers = {

                'User-Agent': random.choice(USER_AGENTS),

                'Accept': 'application/json',

            }

            

            # Search by term

            search_params = {

                'search_terms': query,

                'search_simple': 1,

                'action': 'process',

                'json': 1,

                'page_size': max_products_per_query

            }

            

            response = session.get(

                search_url, 

                params=search_params, 

                headers=headers,

                timeout=10

            )

            

            if response.status_code == 200:

                data = response.json()

                products = data.get('products', [])

                

                for product in products:

                    try:

                        name = product.get('product_name', '').lower()

                        nutri = product.get('nutriments', {})

                        

                        if not name or not nutri:

                            continue

                            

                        grade = score_product(nutri)

                        if grade >= 0.6:

                            facts = [f"{k}:{nutri.get(k, 'N/A')}" for k in 

                                    ["fiber_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g"] 

                                    if k in nutri]

                            desc = f"{name} | {' '.join(facts)}"

                            descriptions.append(desc)

                    except Exception as e:

                        print(f"Error processing product: {e}")

                        continue

            else:

                print(f"API request failed with status code: {response.status_code}")

                

            # Sleep to respect rate limits

            time.sleep(2)

            

        except Exception as e:

            print(f"Error during API request for {query}: {e}")

    

    # Save descriptions to JSONL file for future use

    print(f"Successfully retrieved {len(descriptions)} product descriptions")

    with open("scraped_products.jsonl", "w", encoding="utf-8") as fout:

        for line in descriptions:

            fout.write(json.dumps({"description": line}) + "\n")

            

    return descriptions


def generate_meal_plan(model, vocab, prompt="meal plan:", num_meals=5, portions_per_meal=3):

    """

    Generate a meal plan presented as a table with product names and portions.

    

    Args:

        model: The trained LSTM model

        vocab: Vocabulary dictionary

        prompt: Starting prompt for generation

        num_meals: Number of meals to generate

        portions_per_meal: Number of food items per meal

    

    Returns:

        String containing the tabular meal plan

    """

    # Generate raw text using the model

    raw_plan = generate(model, vocab, prompt, max_len=100)

    print(f"Raw generation: {raw_plan}")

    

    # Parse the generated text to extract food items

    # In real application, we'd implement more sophisticated parsing

    tokens = tokenize(raw_plan)

    

    # Create a list of potential food items (filter out common words, numbers, etc.)

    common_words = {'meal', 'plan', 'for', 'with', 'and', 'the', 'a', 'in', 'of', 'to', 'gdm', 'breakfast', 'lunch', 'dinner', 'snack'}

    food_items = [word for word in tokens if word not in common_words and len(word) > 2]

    

    # If we didn't get enough items, use a fallback list

    fallback_items = [

        "greek yogurt", "quinoa", "lentils", "sweet potato", "tofu",

        "brown rice", "spinach", "chickpeas", "broccoli", "oats",

        "almonds", "avocado", "cottage cheese", "eggs", "blueberries"

    ]

    

    if len(food_items) < num_meals * portions_per_meal:

        # Add some items from fallback

        needed = num_meals * portions_per_meal - len(food_items)

        food_items.extend(fallback_items[:needed])

    

    # Shuffle the food items for variety

    import random

    random.shuffle(food_items)

    

    # Generate portion sizes (in grams or cups)

    portions = {

        "yogurt": "1 cup", "quinoa": "1/2 cup cooked", "lentils": "1/2 cup cooked",

        "sweet": "1 medium", "potato": "1 medium", "tofu": "100g",

        "brown": "1/3 cup cooked", "rice": "1/3 cup cooked", "spinach": "2 cups raw",

        "chickpeas": "1/2 cup", "broccoli": "1 cup", "oats": "1/3 cup dry",

        "almonds": "23 nuts (1oz)", "avocado": "1/4 medium", "cottage": "1/2 cup",

        "blueberries": "3/4 cup", "eggs": "1 large", "cheese": "30g"

    }

    

    # Create the meal plan table

    meal_names = ["Breakfast", "Morning Snack", "Lunch", "Afternoon Snack", "Dinner"]

    

    table = "| Meal | Food Item | Portion |\n"

    table += "|------|----------|--------|\n"

    

    for i in range(min(num_meals, len(meal_names))):

        meal = meal_names[i]

        for j in range(portions_per_meal):

            if i * portions_per_meal + j < len(food_items):

                food = food_items[i * portions_per_meal + j]

                

                # Find portion for this food item

                portion = "1 serving"

                for key, value in portions.items():

                    if key in food:

                        portion = value

                        break

                

                if j == 0:  # First item in the meal includes meal name

                    table += f"| {meal} | {food} | {portion} |\n"

                else:

                    table += f"|  | {food} | {portion} |\n"

    

    return table




# === Use database dump if available ===

def load_from_openfoodfacts_dump(dump_path="openfoodfacts-products.jsonl", min_grade=0.6, max_items=15000):

    print(f"Loading data from Open Food Facts dump: {dump_path}")

    

    if not os.path.exists(dump_path):

        print(f"Dump file not found: {dump_path}")

        return []

        

    descriptions = []

    with open(dump_path, 'r', encoding='utf-8') as f:

        for i, line in enumerate(f):

            if i % 1000 == 0:

                print(f"Processed {i} products...")

                

            try:

                product = json.loads(line)

                name = product.get("product_name", "").lower()

                nutri = product.get("nutriments", {})

                

                if not name or not nutri:

                    continue



                grade = score_product(nutri)

                if grade >= min_grade:

                    facts = [f"{k}:{nutri[k]}" for k in 

                            ["fiber_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g"] 

                            if k in nutri]

                    desc = f"{name} | {' '.join(facts)}"

                    descriptions.append(desc)



                if len(descriptions) >= max_items:

                    break

            except Exception as e:

                continue

                

    print(f"Successfully loaded {len(descriptions)} high-quality food descriptions")

    return descriptions



# === Download a small sample dump if needed ===

def download_sample_dump(filename="sample_products.jsonl", count=5000):

    if os.path.exists(filename):

        print(f"Using existing sample file: {filename}")

        return

        

    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.jsonl"

    print(f"Downloading sample from {url}...")

    

    session = get_session()

    headers = {'User-Agent': random.choice(USER_AGENTS)}

    

    try:

        # Stream the download and take the first 'count' lines

        response = session.get(url, headers=headers, stream=True)

        response.raise_for_status()

        

        with open(filename, 'w', encoding='utf-8') as outf:

            lines = 0

            for line in response.iter_lines(decode_unicode=True):

                if lines >= count:

                    break

                if line:

                    outf.write(line + '\n')

                    lines += 1

                    if lines % 100 == 0:

                        print(f"Downloaded {lines} products...")

                        

        print(f"Downloaded {lines} products to {filename}")

    except Exception as e:

        print(f"Error downloading sample dump: {e}")



if __name__ == "__main__":

    sample_queries = ["lentils", "quinoa", "tofu", "almonds", "broccoli", "avocado", "greek yogurt", 

                       "brown rice", "chia", "sweet potato", "whole wheat", "egg", "blueberries", 

                       "oats", "pumpkin seeds", "spinach", "low fat milk", "apples", "beans", 

                       "chickpeas", "zucchini", "brussels sprouts", "cottage cheese", "barley", 

                       "grapes", "turkey", "tomato", "carrot", "cauliflower", "cucumber"]

    

    categories = ["vegetables", "legumes", "nuts", "breakfast-cereals", "yogurts", "meat-substitutes"]

    

    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", default="sample_products.jsonl", help="Path to Open Food Facts data dump")

    parser.add_argument("--use_api", action="store_true", help="Use API instead of local dump")

    parser.add_argument("--download_sample", action="store_true", help="Download a sample dump first")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")

    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")

    args = parser.parse_args()

    

    descriptions = []

    

    # Get data - try different methods in order

    if args.download_sample:

        download_sample_dump()

    

    if args.use_api:

        descriptions = fetch_from_openfoodfacts_api(sample_queries)

    

    if not descriptions and os.path.exists(args.dump):

        descriptions = load_from_openfoodfacts_dump(args.dump)

    

    if not descriptions:

        # Fallback to a few hardcoded examples so code can still run

        print("Using fallback examples as no data was loaded")

        descriptions = [

            "greek yogurt | fiber_100g:0 carbohydrates_100g:3.6 sugars_100g:3.2 proteins_100g:9.0",

            "quinoa | fiber_100g:7.0 carbohydrates_100g:64.2 sugars_100g:2.8 proteins_100g:14.1",

            "lentils | fiber_100g:8.0 carbohydrates_100g:15.6 sugars_100g:1.8 proteins_100g:24.6",

            "sweet potato | fiber_100g:3.3 carbohydrates_100g:20.1 sugars_100g:4.2 proteins_100g:1.6",

            "tofu | fiber_100g:0.3 carbohydrates_100g:1.9 sugars_100g:0.5 proteins_100g:8.1",

            "brown rice | fiber_100g:3.5 carbohydrates_100g:77.2 sugars_100g:0.8 proteins_100g:7.9",

            "spinach | fiber_100g:2.2 carbohydrates_100g:3.6 sugars_100g:0.4 proteins_100g:2.9",

            "chickpeas | fiber_100g:7.6 carbohydrates_100g:27.4 sugars_100g:2.1 proteins_100g:9.0",

            "broccoli | fiber_100g:2.6 carbohydrates_100g:6.6 sugars_100g:1.7 proteins_100g:2.8",

            "oats | fiber_100g:10.6 carbohydrates_100g:66.3 sugars_100g:1.0 proteins_100g:16.9"

        ]

    

    print(f"Working with {len(descriptions)} high-quality food descriptions.")

    

    vocab = build_vocab(descriptions)

    dataset = ProductDataset(descriptions, vocab)

    if len(dataset) == 0:

        raise ValueError("No data found. Ensure descriptions were loaded properly.")

    

    batch_size = min(args.batch_size, len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    model = LSTMTextGen(len(vocab))

    train(model, dataloader, len(vocab), epochs=args.epochs)



    os.makedirs("./models", exist_ok=True)

    torch.save(model.state_dict(), "./models/off_lstm_model.pt")

    with open("./models/off_vocab.json", "w") as f:

        json.dump(vocab, f)
    meal_plan = generate_meal_plan(model, vocab, "meal plan for gestational diabetes:", num_meals=3, portions_per_meal=3)

    print(meal_plan)



    print("\nModel saved to ./models/off_lstm_model.pt")

    print("Generated Example:")

    print(generate(model, vocab, "meal plan for gdm:"))
