# LSTM meal planner using Open Food Facts JSONL with nutrition-based logistic scoring



import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from collections import Counter

import re

import json

import os

import math



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

                fout.write(json.dumps({"description": line}) + " ")

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



# === Main Run ===

import urllib.request



import requests



from bs4 import BeautifulSoup



def fetch_openfoodfacts_by_category(categories, max_products_per_cat=100, pages=2, delay=1.0):

    descriptions = []

    visited = set()

    search_base = "http://world.openfoodfacts.org/cgi/search.pl"

    for category in categories:

        for page in range(1, pages + 1):

            try:

                params = {

                    "search_terms": category,

                    "search_simple": 1,

                    "action": "process",

                    "page": page,

                    "json": 0

                }

                print(f"Searching: {search_base} with {params}")

                res = requests.get(search_base, params=params)

                soup = BeautifulSoup(res.text, 'html.parser')

                links = soup.select('ul.products li a')

                for link in links[:max_products_per_cat]:

                    href = link.get('href')

                    if not href or not href.startswith("/product/") or href in visited:

                        continue

                    visited.add(href)

                    product_url = f"https://world.openfoodfacts.org{href}"

                    prod_res = requests.get(product_url)

                    prod_soup = BeautifulSoup(prod_res.text, 'html.parser')

                    name_tag = prod_soup.find('h1')

                    name = name_tag.text.strip().lower() if name_tag else None

                    nutri = {}

                    table = prod_soup.select_one('#nutrition_data_table')

                    if table:

                        for row in table.select('tr'):

                            cells = row.find_all('td')

                            if len(cells) >= 2:

                                key = cells[0].text.strip().lower().replace(' ', '_')

                                val = ''.join(filter(str.isdigit, cells[1].text.strip().replace(',', '.')))

                                try:

                                    nutri[key] = float(val)

                                except:

                                    continue

                    if name and nutri:

                        grade = score_product(nutri)

                        if grade >= 0.6:

                            facts = [f"{k}:{nutri[k]}" for k in ["fiber_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g"] if k in nutri]

                            descriptions.append(f"{name} | {' '.join(facts)}")

                    time.sleep(delay)

            except Exception as e:

                print(f"Error scraping search page for {category} {page}: {e}")

    return descriptions



if __name__ == "__main__":

    sample_queries = ["lentils", "quinoa", "tofu", "almonds", "broccoli", "avocado", "greek yogurt", "brown rice", "chia", "sweet potato", "whole wheat", "egg", "blueberries", "oats", "pumpkin seeds", "spinach", "low fat milk", "apples", "beans", "chickpeas", "zucchini", "brussels sprouts", "cottage cheese", "barley", "grapes", "turkey", "tomato", "carrot", "cauliflower", "cucumber"]

    categories = ["vegetables", "legumes", "nuts", "breakfast-cereals", "yogurts", "meat-substitutes"]

    descriptions = fetch_openfoodfacts_by_category(sample_queries)

    print(f"Loaded {len(descriptions)} high-quality food descriptions.")



    vocab = build_vocab(descriptions)

    dataset = ProductDataset(descriptions, vocab)

    dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True) if len(dataset) > 0 else None



    model = LSTMTextGen(len(vocab))

    train(model, dataloader, len(vocab), epochs=3)



    os.makedirs("./models", exist_ok=True)

    torch.save(model.state_dict(), "./models/off_lstm_model.pt")

    with open("./models/off_vocab.json", "w") as f:

        json.dump(vocab, f)



    print("\nModel saved to ./models/off_lstm_model.pt")

    print("Generated Example:")

    print(generate(model, vocab, "meal plan for gdm:"))


