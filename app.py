import os
import logging
import ssl
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# === ENV ===
load_dotenv()
CLIENT_ID = os.getenv("KROGER_CLIENT_ID")
CLIENT_SECRET = os.getenv("KROGER_CLIENT_SECRET")
TOKEN_URL = "https://api.kroger.com/v1/connect/oauth2/token"
PRODUCTS_URL = "https://api.kroger.com/v1/products"
LOCATION_URL = "https://api.kroger.com/v1/locations"
WATCHED_IDS = ["0001111041700"]
POLL_INTERVAL_MINUTES = 1

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Flask & DB ===
db = SQLAlchemy()
scheduler = BackgroundScheduler()

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers('HIGH:!DH:!aNULL')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

class Product(db.Model):
    __tablename__ = "products"
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    brand = db.Column(db.String)
    category = db.Column(db.String)
    image_url = db.Column(db.String)
    product_url = db.Column(db.String)
    regular_price = db.Column(db.Float)
    promo_price = db.Column(db.Float)
    fulfillment = db.Column(db.JSON)
    stock_level = db.Column(db.String)
    size = db.Column(db.String)
    sold_by = db.Column(db.String)
    location = db.Column(db.JSON)
    dimensions = db.Column(db.JSON)
    temperature_sensitive = db.Column(db.Boolean)

class PriceHistory(db.Model):
    __tablename__ = "price_history"
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String, db.ForeignKey("products.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    promo_price = db.Column(db.Float, nullable=False)
    regular_price = db.Column(db.Float, nullable=False)
    product = db.relationship("Product", backref="history")

def get_access_token():
    session = requests.Session()
    session.mount("https://", SSLAdapter())
    payload = {"grant_type": "client_credentials", "scope": "product.compact"}

    try:
        resp = session.post(TOKEN_URL, auth=(CLIENT_ID, CLIENT_SECRET), data=payload)
        resp.raise_for_status()
        token = resp.json()["access_token"]
        if not token:
            raise ValueError("No access_token in response")
        logger.info("✅ Obtained access token")
        return token
    except Exception as e:
        logger.error(f"❌ Token error: {e}")
        raise

def fetch_nearest_location(token, zip_code="45202"):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"filter.zipCode.near": zip_code, "filter.limit": 1}
    resp = requests.get(LOCATION_URL, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0] if data else {}

def fetch_products(token, term, limit=50, location_id=None):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"filter.term": term, "filter.limit": limit}
    if location_id:
        params["filter.locationId"] = location_id

    products, next_url = [], PRODUCTS_URL
    while next_url:
        try:
            resp = requests.get(next_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            products.extend(data.get("data", []))
            link = resp.headers.get("Link", "")
            next_url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    next_url = part.split(";")[0].strip()[1:-1]
                    break
            params = {}  # Clear params after first request
        except Exception as e:
            logger.error(f"❌ Product fetch error: {e}")
            break
    return products

def map_kroger_to_zenday(data):
    item = data.get("items", [{}])[0]
    aisle = (data.get("aisleLocations") or [{}])[0]
    image = data.get("images", [{}])[0].get("sizes", [{}])[0]
    return {
        "id": data.get("productId"),
        "name": data.get("description"),
        "brand": data.get("brand"),
        "category": data.get("categories", [None])[0],
        "image_url": image.get("url"),
        "product_url": f"https://www.kroger.com{data.get('productPageURI')}",
        "price": {
            "regular": item.get("price", {}).get("regular"),
            "promo": item.get("price", {}).get("promo"),
        },
        "fulfillment": item.get("fulfillment", {}),
        "stock_level": item.get("inventory", {}).get("stockLevel"),
        "size": item.get("size"),
        "sold_by": item.get("soldBy"),
        "location": {
            "aisle": aisle.get("number"),
            "shelf": aisle.get("shelfNumber"),
            "bay": aisle.get("bayNumber"),
            "side": aisle.get("side"),
        },
        "dimensions": {
            "width": float(data.get("itemInformation", {}).get("width", 0)),
            "height": float(data.get("itemInformation", {}).get("height", 0)),
            "depth": float(data.get("itemInformation", {}).get("depth", 0)),
        },
        "temperature_sensitive": data.get("temperature", {}).get("heatSensitive", False),
    }

def process_product_data(prod_data):
    pid = prod_data["id"]
    new_reg, new_pr = prod_data["price"]["regular"], prod_data["price"]["promo"]
    existing = Product.query.get(pid)

    if existing:
        old_pr = existing.promo_price or 0
        if new_pr is not None and new_pr < old_pr:
            existing.regular_price = new_reg
            existing.promo_price = new_pr
            db.session.add(existing)
            db.session.add(PriceHistory(product_id=pid, promo_price=new_pr, regular_price=new_reg))
            db.session.commit()
            print(f"🔔 Price drop for {pid}: {old_pr} → {new_pr}")
            return {"alert": True, "old_price": old_pr, "new_price": new_pr}
        db.session.add(PriceHistory(product_id=pid, promo_price=new_pr, regular_price=new_reg))
        db.session.commit()
        return {"alert": False}

    new_p = Product(
        id=pid, name=prod_data["name"], brand=prod_data["brand"],
        category=prod_data["category"], image_url=prod_data["image_url"],
        product_url=prod_data["product_url"], regular_price=new_reg, promo_price=new_pr,
        fulfillment=prod_data["fulfillment"], stock_level=prod_data["stock_level"],
        size=prod_data["size"], sold_by=prod_data["sold_by"],
        location=prod_data["location"], dimensions=prod_data["dimensions"],
        temperature_sensitive=prod_data["temperature_sensitive"]
    )
    db.session.add(new_p)
    db.session.add(PriceHistory(product_id=pid, promo_price=new_pr, regular_price=new_reg))
    db.session.commit()
    print(f"🆕 New product added: {pid}")
    return {"alert": True, "new_price": new_pr}

def monitor_watched_products():
    with app.app_context():
        token = get_access_token()
        loc_id = fetch_nearest_location(token, zip_code="45202").get("locationId")
        if not loc_id:
            print("⚠️ No Kroger location found")
            return
        for pid in WATCHED_IDS:
            items = fetch_products(token, term=pid, limit=5, location_id=loc_id)
            raw = next((i for i in items if i.get("productId") == pid), None)
            if raw:
                process_product_data(map_kroger_to_zenday(raw))
            else:
                print(f"⚠️ No data for {pid}")

# === Flask App ===
def create_app():
    global app
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///zenday.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()

    scheduler.add_job(monitor_watched_products, "interval", minutes=POLL_INTERVAL_MINUTES, id="polling")
    scheduler.start()

    @app.route("/")
    def home():
        return "✅ Zenday Kroger Tracker Running"

    @app.route("/products", methods=["GET"])
    def list_products():
        return jsonify([
            {
                "id": p.id, "name": p.name, "brand": p.brand,
                "category": p.category, "regular_price": p.regular_price,
                "promo_price": p.promo_price, "stock_level": p.stock_level
            } for p in Product.query.all()
        ])

    @app.route("/product/<product_id>/history", methods=["GET"])
    def get_price_history(product_id):
        history = PriceHistory.query.filter_by(product_id=product_id).order_by(PriceHistory.timestamp.desc()).all()
        return jsonify([
            {
                "timestamp": h.timestamp.isoformat(),
                "promo_price": h.promo_price,
                "regular_price": h.regular_price,
            } for h in history
        ])

    return app

# === Main Entrypoint ===
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
