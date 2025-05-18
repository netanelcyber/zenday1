# Lightweight PyTorch LSTM training script for meal planning
# Adapted using dietary guidance for GDM from PMID:33677540 with Food.com-style recipe data (Spoonacular API)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import json
import os
import random
import requests

# === Spoonacular API Integration ===
def fetch_spoonacular_recipes(query="diabetic", number=100, api_key="1db091aa2f084dfdb05116ef53f9db1e"):
    url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": query,
        "number": number,
        "addRecipeInformation": True,
        "apiKey": api_key
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        return res.json().get("results", [])
    else:
        print("Spoonacular API error:", res.status_code, res.text)
        return []

# === Load recipe data ===
def load_descriptions_spoonacular(api_data, min_score=2):
    descriptions = []
    gdm_keywords = [item.get("title", "").lower() for item in api_data if item.get("title")]

    for item in api_data or []:
        title = item.get("title", "")
        summary = re.sub('<[^<]+?>', '', item.get("summary", ""))
        nutrients = item.get("nutrition", {}).get("nutrients", [])
        facts = [f"{n['name']}:{n['amount']}{n['unit']}" for n in nutrients if n.get("amount")]
        nutrition_score = 0
        for n in nutrients:
            name = n.get("name", "").lower()
            value = n.get("amount", 0)
            if "fiber" in name and value >= 3:
                nutrition_score += 1
            if "carbohydrate" in name and value <= 30:
                nutrition_score += 1
            if "sugar" in name and value <= 5:
                nutrition_score += 1
            if "protein" in name and value >= 8:
                nutrition_score += 1

        full_desc = f"{title} | {summary} | {' '.join(facts)}".lower()
        if nutrition_score >= min_score and any(kw in full_desc for kw in gdm_keywords):
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
    SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")
    recipe_data = fetch_spoonacular_recipes(query="gestational diabetes", api_key="1db091aa2f084dfdb05116ef53f9db1e")

    descriptions = load_descriptions_spoonacular(recipe_data)
    print(f"Loaded {len(descriptions)} Spoonacular recipe descriptions.")
    if not descriptions:
        raise ValueError("No recipe descriptions found from Spoonacular.")

    vocab = build_vocab(descriptions)
    dataset = ProductDataset(descriptions, vocab)
    if len(dataset) == 0:
        raise ValueError("No valid data for training.")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMTextGen(len(vocab))
    train(model, dataloader, len(vocab), epochs=3)

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/spoonacular_gdm_lstm_model.pt")
    with open("./models/spoonacular_gdm_vocab.json", "w") as f:
        json.dump(vocab, f)

    print("\nModel saved to ./models/spoonacular_gdm_lstm_model.pt")
    print("Generated Example:")
    print(generate(model, vocab, "meal plan for gdm:"))
