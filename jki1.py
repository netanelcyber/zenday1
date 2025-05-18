# Lightweight PyTorch LSTM training script for meal planning
# Adapted using dietary guidance for GDM from PMID:33677540 with automatic product + nutrition download

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import json
import os
import random
import requests

# === Kroger API Integration ===
def fetch_kroger_products(query="", limit=100, token=""):  # Provide token or handle externally
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "filter.term": query,
        "filter.limit": limit,
        "filter.locationId": "01400943"  # use a default Kroger location or update dynamically
    }
    res = requests.get("https://api.kroger.com/v1/products", headers=headers, params=params)
    if res.status_code == 200:
        return res.json().get("data", [])
    else:
        print("Failed to fetch products:", res.status_code, res.text)
        return []

# === Load product descriptions from JSON files or API ===
def load_descriptions(directory="./data", api_data=None):
    descriptions = []
    gdm_keywords = ["low glycemic", "high fiber", "whole grain", "unsweetened", "lean protein", "nonfat", "sugar free", "no sugar added"]

    # Load from files
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        api_data = (api_data or []) + data.get("data", [])
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    # Parse API or local data
    for item in api_data or []:
        desc = item.get("description", "").strip().lower()
        nutrition = item.get("nutritional", {})
        facts = []
        for key in ["calories", "carbohydrates", "sugars", "fiber", "protein", "totalFat"]:
            if key in nutrition:
                facts.append(f"{key}:{nutrition[key]}")
        full_desc = desc + " | " + " ".join(facts) if facts else desc
        if desc and any(kw in desc for kw in gdm_keywords):
            descriptions.append(full_desc)

    return descriptions[:15000]

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
if __name__ == "__main__":
    # === Optional: replace with real API token ===
    TOKEN = os.getenv("KROGER_TOKEN", "")  # Or paste your Bearer token here
    downloaded_data = fetch_kroger_products(query="diabetic", limit=100, token=TOKEN)

    descriptions = load_descriptions("./data", api_data=downloaded_data)
    print(f"Loaded {len(descriptions)} product descriptions.")
    if not descriptions:
        raise ValueError("No product descriptions found.")

    vocab = build_vocab(descriptions)
    dataset = ProductDataset(descriptions, vocab)
    if len(dataset) == 0:
        raise ValueError("No valid data for training.")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMTextGen(len(vocab))
    train(model, dataloader, len(vocab), epochs=3)

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/gdm_lstm_model.pt")
    with open("./models/gdm_vocab.json", "w") as f:
        json.dump(vocab, f)

    print("\nModel saved to ./models/gdm_lstm_model.pt")
    print("Generated Example:")
    print(generate(model, vocab, "meal plan for gdm:"))
