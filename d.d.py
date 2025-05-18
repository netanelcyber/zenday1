# DNN Text Generator (Char-level) using NumPy
# Trained on PubMed abstracts about Gestational Diabetes

import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from Bio import Entrez

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
Entrez.email = "nsh531@gmail.com"

# === Parameters ===
SEQ_LEN = 10
HIDDEN_SIZE = 256
LR = 0.006
EPOCHS = 120
BATCH_SIZE = 64

# === Data Preparation ===
def fetch_pubmed_text():
    print("[INFO] Fetching abstracts from PubMed...")
    search = Entrez.esearch(db="pubmed", term="gestational diabetes", retmax=50)
    ids = Entrez.read(search)["IdList"]
    fetch = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
    abstracts = fetch.read().strip().split("\n\n")
    text = " ".join([a for a in abstracts if len(a) > 50]).lower()
    text = re.sub(r"[^a-z .,;!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Create dataset ===
def create_dataset(text, seq_len):
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    X, y = [], []
    for i in range(len(text) - seq_len):
        X.append([char2idx[ch] for ch in text[i:i+seq_len]])
        y.append(char2idx[text[i+seq_len]])
    return np.array(X), np.array(y), char2idx, idx2char

# === DNN Model ===
def init_dnn(input_dim, hidden_dim, output_dim):
    model = {
        "W1": np.random.randn(hidden_dim, input_dim) * 0.01,
        "b1": np.zeros((hidden_dim, 1)),
        "W2": np.random.randn(output_dim, hidden_dim) * 0.01,
        "b2": np.zeros((output_dim, 1)),
    }
    return model

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def one_hot(index, size):
    vec = np.zeros((size, 1))
    vec[index] = 1
    return vec

def forward(model, x):
    z1 = np.dot(model["W1"], x) + model["b1"]
    a1 = relu(z1)
    z2 = np.dot(model["W2"], a1) + model["b2"]
    y_hat = softmax(z2)
    return z1, a1, z2, y_hat

def backward(model, x, y, z1, a1, y_hat):
    dy = y_hat - y  # Cross-entropy grad
    dW2 = np.dot(dy, a1.T)
    db2 = dy
    da1 = np.dot(model["W2"].T, dy)
    dz1 = da1 * relu_deriv(z1)
    dW1 = np.dot(dz1, x.T)
    db1 = dz1
    return dW1, db1, dW2, db2

def train(model, X, y, vocab_size, epochs=10):
    for epoch in range(epochs):
        loss_total = 0
        for i in range(len(X)):
            x_seq = X[i]
            target = y[i]
            x_input = np.concatenate([one_hot(ix, vocab_size) for ix in x_seq], axis=0)
            y_target = one_hot(target, vocab_size)

            z1, a1, z2, y_hat = forward(model, x_input)
            loss = -np.log(y_hat[target, 0])
            loss_total += loss

            dW1, db1, dW2, db2 = backward(model, x_input, y_target, z1, a1, y_hat)
            model["W1"] -= LR * dW1
            model["b1"] -= LR * db1
            model["W2"] -= LR * dW2
            model["b2"] -= LR * db2

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_total/len(X) :.4f}")

# === Generation ===
def generate(model, seed, char2idx, idx2char, vocab_size, length=300):
    result = seed
    input_seq = [char2idx[ch] for ch in seed]
    for _ in range(length):
        x_input = np.concatenate([one_hot(ix, vocab_size) for ix in input_seq], axis=0)
        _, _, _, y_hat = forward(model, x_input)
        next_ix = np.random.choice(range(vocab_size), p=y_hat.ravel())
        next_char = idx2char[next_ix]
        result += next_char
        input_seq = input_seq[1:] + [next_ix]
    return result

# === NLP Filter ===
def extract_valid_sentence(text):
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        if len(tokens) > 4:
            if any(tag.startswith("NN") for _, tag in tags) and any(tag.startswith("VB") for _, tag in tags):
                return sentence
    return "(No valid sentence found)"

# === Main Flow ===
if __name__ == "__main__":
    text = fetch_pubmed_text()
    X, y, char2idx, idx2char = create_dataset(text, SEQ_LEN)
    vocab_size = len(char2idx)
    input_dim = vocab_size * SEQ_LEN
    model = init_dnn(input_dim, HIDDEN_SIZE, vocab_size)
    train(model, X[:10000], y[:10000], vocab_size, epochs=EPOCHS)
    raw = generate(model, seed="glucose is", char2idx=char2idx, idx2char=idx2char, vocab_size=vocab_size)
    final_sentence = extract_valid_sentence(raw)
    print("\n🧠 Generated Sentence:\n", final_sentence)

