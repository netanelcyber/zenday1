import logging
import platform
import time
import base64
import requests
import urllib.parse
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# === OAuth2 Configuration ===
CLIENT_ID = "tester3-bbc58vmv"
CLIENT_SECRET = "16KeKNgOqBiz_KK_Y7I74mtvAgZv7QsHndYzJ7gm"
REDIRECT_URI = "https://185.181.8.11/x"  # Use HTTPS in production
TOKEN_URL = "https://api.kroger.com/v1/connect/oauth2/token"

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kroger_cart_api")

# === FastAPI App ===
app = FastAPI(
    title="Kroger Cart API (Customer Context)",
    version="1.0.0",
    description="OAuth2 flow using authorization code grant for customer-specific Kroger cart actions."
)

# === In-memory status ===
operation_status = {"status": "idle", "message": "", "timestamp": ""}
@app.post("/kroger/token")
def get_access_token():
    import base64
    CLIENT_ID = "tester3-bbc58vmv"
    CLIENT_SECRET = "16KeKNgOqBiz_KK_Y7I74mtvAgZv7QsHndYzJ7gm"

    payload = {
        "grant_type": "client_credentials",
        "scope": "product.compact"
    }

    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        resp = requests.post(TOKEN_URL, headers=headers, data=payload)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise ValueError("No access_token found in response")
        logging.info("Access token obtained successfully")
        print(token)
        return token
    except Exception as e:
        logging.error(f"Failed to get access token: {e}")
        raise

def update_status(status: str, message: str):
    operation_status.update({
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    logger.info(f"[{status.upper()}] {message}")

# === Models ===
class CartItem(BaseModel):
    upc: str
    quantity: int = 1

class AddToCartRequest(BaseModel):
    access_token: str
    items: List[CartItem]

# === OAuth2 Flow ===
@app.get("/kroger/authorize")
def generate_authorization_url(state: str = "xyz"):
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "cart.basic:write product.compact customer.profile.basic",
        "state": state
    }
    url = f"https://api.kroger.com/v1/connect/oauth2/authorize?{urllib.parse.urlencode(params)}"
    return {"authorize_url": url, "token_url": TOKEN_URL}

@app.get("/callback")
def oauth_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code parameter")
    token = exchange_code_for_token(code)
    return {"token_response": token}

def exchange_code_for_token(code: str):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error("Token exchange failed: %s", response.text)
        raise HTTPException(status_code=response.status_code, detail=response.text)

# === Cart API ===
@app.post("/kroger/cart/add")
def add_to_cart(req: AddToCartRequest):
    update_status("starting", "Adding items to cart with customer context")
    url = "https://api.kroger.com/v1/cart/add"
    headers = {
        "Authorization": f"Bearer {req.access_token}",
        "Content-Type": "application/json"
    }
    payload = {"items": [item.dict() for item in req.items]}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        update_status("success", "Items added to cart")
        return response.json()
    else:
        update_status("error", response.text)
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/kroger/status")
def get_status():
    return operation_status

@app.post("/kroger/reset")
def reset_status():
    update_status("idle", "Status reset")
    return {"message": "Status reset"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/x")
def receive_x():
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code parameter")
    token = exchange_code_for_token(code)
    return {"token_response": token}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("x:app", host="0.0.0.0", port=8050, reload=True)
