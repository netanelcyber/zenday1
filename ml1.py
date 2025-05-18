
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np, faiss, os, requests, json, re, pickle



import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # תרגומים נוספים (לא חובה)


from nltk.corpus import wordnet




app = Flask(__name__)
os.makedirs("uploaded_jsons", exist_ok=True)

# === Embedding Setup ===
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

# === FAISS Vector Search Setup ===
dimension = 384
index = faiss.IndexFlatL2(dimension)
index_is_empty = True
product_metadata = []

# === Kroger Auth Setup ===
URL = 'https://api.kroger.com/v1/connect/oauth2/token'
CREDS = 'Basic emVuZGF5Mi0yNDMyNjEyNDMwMzQyNDM4NGU3NTRiMzA0Mjc0Mzg3MTM0NGY3MjM0NDI0ODU3NWE0MzRiMzc1NzJlNDI0MTZhNTM0OTY0NWEzNTQ4NzAzNTRhNGQ1NjYzNmY3MDMxNGUzNzcwNGQ2YzZiNTk3MTVhNjI3MzY0NGY0OTYwMTgzNDIzMDA3MzIzMzA4OmwxYkpPMXhHbEpKQm85bGNoY2xOaVhPWnRIRjNRT2FrV2FRdEJPMnI='  # Replace with your actual client_id:client_secret (Base64)
CT = "application/x-www-form-urlencoded"
token = requests.post(URL, headers={'Content-Type': CT, "Authorization": CREDS},
                      data={"grant_type": "client_credentials", "scope": "product.compact"}).json()["access_token"]
KROGER_HEADERS = {"Authorization": f"Bearer {token}"}
print(token)
# === RNN Setup (Pure NumPy) ===
hidden_size, seq_length, learning_rate = 100, 250, 1e-4
meal_data, chars, vocab_size = "tofu wrap", [], 0
char_to_ix, ix_to_char = {}, {}
Wxh = Whh = Why = bh = by = None

def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1)); x[seed_ix] = 1; ixes = []
    for _ in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1)); x[ix] = 1
        ixes.append(ix)
    return ''.join(ix_to_char[ix] for ix in ixes)

def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}; hs[-1] = np.copy(hprev); loss = 0
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)); xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])
    return loss, xs, hs, ps

def train_rnn(iters=500):
    global Wxh, Whh, Why, bh, by
    data_ix = [char_to_ix[ch] for ch in meal_data]
    hprev = np.zeros((hidden_size, 1))
    for n in range(iters):
        if n + seq_length + 1 >= len(data_ix): n = 0
        inputs = data_ix[n:n + seq_length]
        targets = data_ix[n + 1:n + seq_length + 1]
        loss, xs, hs, ps = lossFun(inputs, targets, hprev)
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by); dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t]); dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T); dby += dy
            dh = np.dot(Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby
        if n % 10 == 0:
            print(f"[RNN] Iter {n} Loss: {loss}")
            print(sample(hprev, inputs[0], 100))
    with open("rnn_model.pkl", "wb") as f:
        pickle.dump((Wxh, Whh, Why, bh, by), f)



@app.route("/mealplan", methods=["POST"])
def generate_meal_plan():
    goal = request.form.get("goal", "t").strip().lower()
    try:
        with open("rnn_model.pkl", "rb") as f:
            global Wxh, Whh, Why, bh, by
            Wxh, Whh, Why, bh, by = pickle.load(f)
    except:
        return "⚠️ Model not trained. Use /scrape_kroger_train_rnn first."

    seed_ix = char_to_ix.get(goal[0], 0)
    raw_text = sample(np.zeros((hidden_size, 1)), seed_ix, 400)
    lines = raw_text.split("\n")
    valid_lines = []

    for line in lines:
        tokens = re.findall(r"[a-z]+", line.lower())
        valid = [word for word in tokens if wordnet.synsets(word)]  # רק מילים שיש להן ערך ב-WordNet
        if len(valid) >= max(1, len(tokens) // 2):
            valid_lines.append(" ".join(valid))

    final = "\n".join(valid_lines or ["(no valid meal lines)"])
    return f"<h2>Meal Plan (Filtered by WordNet)</h2><pre>{final}</pre><p><a href='/'>Back</a></p>"



from Bio import Entrez
Entrez.email = "nsh531@gmail.com"

@app.route("/pubmed_train")
def fetch_pubmed_and_train_rnn():
    global meal_data, chars, vocab_size, char_to_ix, ix_to_char, Wxh, Whh, Why, bh, by

    try:
        # שלב 1: חיפוש מאמרים שזמינים ב-PMC
        search_handle = Entrez.esearch(db="pubmed", term="gestational diabetes",
                                       retmax=100, sort="relevance")
        search_results = Entrez.read(search_handle)
        ids = search_results.get("IdList", [])
        if not ids:
            return "<h2>No PMC articles found for GDM</h2>"

        # שלב 2: שליפת תקצירים
        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
        abstracts = fetch_handle.read().strip().split("\n\n")

        # שלב 3: הכנת טקסט ללימוד
        combined_text = "\n".join([a for a in abstracts if len(a) > 50])
        if not combined_text:
            return "<h2>No valid abstracts to train on.</h2>"

        meal_data = combined_text.lower()
        chars = sorted(set(meal_data))
        vocab_size = len(chars)
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # שלב 4: אתחול מחדש של המודל
        Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        Why = np.random.randn(vocab_size, hidden_size) * 0.01
        bh = np.zeros((hidden_size, 1))
        by = np.zeros((vocab_size, 1))

        # שלב 5: אימון על התקצירים
        train_rnn(iters=150)

        return f"<h2>✅ Trained on {len(abstracts)} PubMed GDM abstracts</h2>" \
               f"<p><a href='/mealplan'>Generate Meal Plan from New Model</a></p>" \
               f"<p><a href='/'>Back</a></p>"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h3>Error: {e}</h3><a href='/'>Back</a>"


@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>GDM Meal Planner</title>
    <style>
        body { font-family: Arial; padding: 20px; max-width: 700px; margin: auto; }
        h2, h3 { color: #2c3e50; }
        form { margin-bottom: 20px; }
        input[type="text"] { padding: 8px; width: 80%; max-width: 400px; }
        button { padding: 8px 16px; margin-top: 5px; }
        .section { border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
    </style>
</head>
<body>

    <h1>🍽️ GDM Meal Planner Dashboard</h1>

    <div class="section">
        <h2>1. 🔄 Scrape & Train from Kroger</h2>
        <p>Fetch product names from Kroger and train the RNN model.</p>
        <form method="get" action="/scrape_kroger_train_rnn">
            <button type="submit">🚀 Scrape Kroger + Train RNN</button>
        </form>
    </div>

<div class="section">
    <h2>🧬 Train RNN on PubMed Articles (GDM)</h2>
    <form method="get" action="/pubmed_train">
        <button>📘 Fetch & Train from PubMed</button>
    </form>
</div>



    <div class="section">
        <h2>2. 🧠 Generate Meal Plan</h2>
        <p>Enter the starting letter or word for meal generation.</p>
        <form method="post" action="/mealplan">
            <input name="goal" placeholder="Start with letter (e.g. t)" required>
            <br><button type="submit">🍲 Generate Plan</button>
        </form>
    </div>

    <div class="section">
        <h2>3. 🧪 Manual Train on Default Data</h2>
        <p>Train RNN on fixed internal data for testing.</p>
        <form method="get" action="/train_rnn">
            <button type="submit">🧪 Train RNN (static)</button>
        </form>
    </div>

</body>
</html>
    """)



@app.route("/a")
def index8():
    return render_template_string("""
    <html><body><h2>GDM Meal Planner</h2>
    <form method="get" action="/scrape_kroger_train_rnn"><button>Scrape Kroger + Train RNN</button></form>
    <form method="post" action="/mealplan"><input name="goal" placeholder="Start letter" required><button>Generate Plan</button></form>
    <form method="get" action="/train_rnn"><button>Train RNN Locally</button></form>
    </body></html>
    """)

@app.route("/train_rnn")
def train_rnn_route():
    train_rnn()
    return "✅ RNN trained on static `meal_data`."

@app.route("/scrape_kroger_train_rnn")
def scrape_kroger_and_train_rnn():
    global meal_data, chars, vocab_size, char_to_ix, ix_to_char, Wxh, Whh, Why, bh, by
    terms = ["tofu", "chicken", "salmon", "quinoa", "pasta", "soup", "bowl", "wrap"]
    headers = {"Authorization": f"Bearer {token}"}
    descs = []
    for term in terms:
        res = requests.get("https://api.kroger.com/v1/products",
            headers=headers, params={"filter.term": term, "filter.limit": 41})
        print(res.json())
        for item in res.json().get("data", []):
            d = item.get("description", "").lower()
            if len(d) > 10: descs.append(d)
    meal_data = "\n".join(descs)
    chars = sorted(list(set(meal_data))); vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}; ix_to_char = {i: ch for i, ch in enumerate(chars)}
    Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01
    Why = np.random.randn(vocab_size, hidden_size) * 0.01
    bh = np.zeros((hidden_size, 1)); by = np.zeros((vocab_size, 1))
    train_rnn(12500)
    return f"✅ Scraped {len(descs)} Kroger items and trained RNN!"

@app.route("/mealplan2", methods=["POST"])
def generate_m2eal_plan():
    goal = request.form.get("goal", "t").strip().lower()
    try:
        with open("rnn_model.pkl", "rb") as f:
            global Wxh, Whh, Why, bh, by
            Wxh, Whh, Why, bh, by = pickle.load(f)
    except FileNotFoundError:
        return "Model not trained yet. Use /scrape_kroger_train_rnn."
    seed_ix = char_to_ix.get(goal[0], 0)
    text = sample(np.zeros((hidden_size, 1)), seed_ix, 300)
    return f"<h2>Meal Plan (Generated):</h2><pre>{text}</pre><p><a href='/'>Home</a></p>"

if __name__ == "__main__":
    print("🚀 App running at http://127.0.0.1:4403")
    app.run(host="0.0.0.0", port=4403)
