# Lightweight PyTorch LSTM training script for meal planning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import json
import os

# === Load product descriptions from JSON files ===
def load_descriptions(directory="/root/uploaded_jsons/"):
    descriptions = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for item in data.get("data", []):
                        desc = item.get("description", "").strip()
                        if desc:
                            descriptions.append(desc)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
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
        self.data = [self.encode(t) for t in texts]

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
    descriptions = load_descriptions()
    vocab = build_vocab(descriptions)
    dataset = ProductDataset(descriptions, vocab)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMTextGen(len(vocab))
    train(model, dataloader, len(vocab), epochs=3)

    print("\nGenerated Example:")
    print(generate(model, vocab, "meal plan for gdm:"))
