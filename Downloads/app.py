from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import platform
import random
import time
import uvicorn

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kroger-automation")

# --- FastAPI app ---
app = FastAPI(
    title="Kroger Automation API",
    description="API to login to Kroger and add products to cart using Chrome or Firefox",
    version="1.0.0"
)

# --- Proxy list ---
PROXY_LIST = [
    {"ip": "8.211.194.78", "port": "9080"},
    {"ip": "8.137.13.191", "port": "9098"},
    {"ip": "8.138.131.110", "port": "3128"},
    {"ip": "47.76.144.139", "port": "9080"},
    {"ip": "8.211.195.173", "port": "9080"},
    {"ip": "47.122.56.158", "port": "3128"},
    {"ip": "47.104.28.135", "port": "8081"},
    {"ip": "47.91.29.151", "port": "6379"},
    {"ip": "8.137.13.191", "port": "3128"},
    {"ip": "35.154.78.253", "port": "80"},
    {"ip": "35.154.78.253", "port": "1080"},
    {"ip": "149.129.255.179", "port": "4002"},
    {"ip": "149.129.226.9", "port": "9160"},
    {"ip": "39.102.208.23", "port": "8888"},
    {"ip": "8.220.136.174", "port": "4006"},
    {"ip": "8.220.136.174", "port": "5060"}
]

def get_random_proxy() -> Dict[str, str]:
    return random.choice(PROXY_LIST)

# --- Models ---
class Credentials(BaseModel):
    email: str = "nsh531@gmail.com"
    password: str = "318962420A"

class KrogerRequest(BaseModel):
    credentials: Optional[Credentials] = Credentials()
    product: str = "milk"
    headless: bool = False
    use_proxy: bool = False
    browser: str = "chrome"  # "chrome" or "firefox"

# --- Status ---
operation_status = {
    "chrome": {"status": "idle", "message": "", "timestamp": ""},
    "firefox": {"status": "idle", "message": "", "timestamp": ""}
}

def update_status(browser: str, status: str, message: str):
    operation_status[browser]["status"] = status
    operation_status[browser]["message"] = message
    operation_status[browser]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{browser}] {status} - {message}")

# --- Automation Task ---
def kroger_automation_task(request: KrogerRequest, browser: str):
    try:
        update_status(browser, "starting", "Setting up webdriver...")

        # Set up browser options
        driver = None
        if browser == "chrome":
            options = webdriver.ChromeOptions()
            if request.headless:
                options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-infobars')
            options.add_argument('--start-maximized')
            options.add_argument('--log-level=3')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36')
            if request.use_proxy:
                selected_proxy = get_random_proxy()
                proxy_address = f"{selected_proxy['ip']}:{selected_proxy['port']}"
                options.add_argument(f'--proxy-server=https://{proxy_address}')
                update_status(browser, "configuring", f"Proxy: {proxy_address}")
            elif platform.system() == 'Windows':
                options.add_argument('--proxy-server="direct://"')
                options.add_argument('--proxy-bypass-list=*')
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

        elif browser == "firefox":
            options = webdriver.FirefoxOptions()
            if request.headless:
                options.add_argument('--headless')
            if request.use_proxy:
                selected_proxy = get_random_proxy()
                proxy_address = f"{selected_proxy['ip']}:{selected_proxy['port']}"
                options.set_preference("network.proxy.type", 1)
                options.set_preference("network.proxy.http", selected_proxy['ip'])
                options.set_preference("network.proxy.http_port", int(selected_proxy['port']))
                options.set_preference("network.proxy.ssl", selected_proxy['ip'])
                options.set_preference("network.proxy.ssl_port", int(selected_proxy['port']))
                update_status(browser, "configuring", f"Proxy: {proxy_address}")
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)

        else:
            update_status(browser, "error", f"Unsupported browser: {browser}")
            return

        driver.set_page_load_timeout(30)

        # --- Kroger automation steps ---
        update_status(browser, "navigating", "Navigating to Kroger's website...")
        driver.get("https://www.kroger.com/signin")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "email")))
        update_status(browser, "logging_in", f"Logging in as {request.credentials.email}...")
        driver.find_element(By.ID, "email").send_keys(request.credentials.email)
        driver.find_element(By.ID, "password").send_keys(request.credentials.password)
        driver.find_element(By.XPATH, "//button[contains(text(), 'Sign In')]").click()

        try:
            WebDriverWait(driver, 15).until(EC.url_contains("/dashboard"))
            update_status(browser, "logged_in", "Successfully logged in!")
        except TimeoutException:
            update_status(browser, "verification_needed", "Login may require additional verification.")
            time.sleep(30)

        update_status(browser, "searching", f"Searching for {request.product}...")
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search']"))
        )
        search_box.clear()
        search_box.send_keys(request.product)
        search_box.submit()

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.ProductCard"))
        )
        update_status(browser, "search_complete", "Search results loaded.")

        try:
            update_status(browser, "adding_to_cart", "Attempting to add product to cart...")
            add_to_cart_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Add to Cart') or contains(text(), 'Add to Cart')]"))
            )
            add_to_cart_button.click()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.kds-Counter"))
            )
            update_status(browser, "success", f"Successfully added {request.product} to cart!")
        except Exception as e:
            update_status(browser, "cart_error", f"Error adding product to cart: {str(e)}")

        driver.quit()
        if operation_status[browser]["status"] not in ["success", "cart_error", "error"]:
            update_status(browser, "completed", "Process completed but with unknown result")
    except Exception as e:
        update_status(browser, "system_error", f"System error: {str(e)}")

# --- Endpoints ---

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/kroger/reset", tags=["Kroger"])
async def reset_status(browser: str = Query("chrome", enum=["chrome", "firefox"])):
    update_status(browser, "idle", "Status reset by user")
    return {"message": f"Status for {browser} reset successfully"}

@app.post("/kroger/add-to-cart", tags=["Kroger"])
async def add_to_cart(request: KrogerRequest, background_tasks: BackgroundTasks):
    browser = request.browser.lower()
    if operation_status[browser]["status"] not in ["idle", "completed", "error", "system_error", "success", "cart_error"]:
        return JSONResponse(
            status_code=409,
            content={
                "message": f"Another automation task is already running in {browser}",
                "current_status": operation_status[browser]
            }
        )
    update_status(browser, "queued", "Task queued and ready to start")
    try:
        background_tasks.add_task(kroger_automation_task, request, browser)
    except Exception as e:
        update_status(browser, "system_error", f"Failed to start background task: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to start automation task: {str(e)}"}
        )
    return {
        "message": f"Kroger automation process started in {browser}",
        "status": "queued",
        "check_status_at": f"/kroger/status?browser={browser}"
    }

@app.get("/kroger/status", tags=["Kroger"])
async def get_status(browser: str = Query("chrome", enum=["chrome", "firefox"])):
    return operation_status[browser]

@app.get("/", tags=["Root"])
async def root():
    return {
        "api": "Kroger Automation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /kroger/add-to-cart": "Start the Kroger login and add-to-cart process (choose browser)",
            "GET /kroger/status": "Check the status of the current operation (choose browser)"
        }
    }

@app.get("/proxies", tags=["Proxies"])
async def list_proxies():
    return {"proxies": PROXY_LIST, "count": len(PROXY_LIST)}

@app.post("/proxies/add", tags=["Proxies"])
async def add_proxy(ip: str, port: str):
    new_proxy = {"ip": ip, "port": port}
    for proxy in PROXY_LIST:
        if proxy["ip"] == ip and proxy["port"] == port:
            return JSONResponse(
                status_code=409,
                content={"message": "Proxy already exists in the list"}
            )
    PROXY_LIST.append(new_proxy)
    return {"message": "Proxy added successfully", "proxy": new_proxy}

@app.delete("/proxies/remove", tags=["Proxies"])
async def remove_proxy(ip: str, port: str):
    proxy_to_remove = {"ip": ip, "port": port}
    if proxy_to_remove in PROXY_LIST:
        PROXY_LIST.remove(proxy_to_remove)
        return {"message": "Proxy removed successfully"}
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "Proxy not found in the list"}
        )

@app.post("/proxies/test", tags=["Proxies"])
async def test_proxy(background_tasks: BackgroundTasks, ip: str, port: str, browser: str = Query("chrome", enum=["chrome", "firefox"])):
    update_status(browser, "testing_proxy", f"Testing proxy {ip}:{port}")
    test_request = KrogerRequest(
        headless=True,
        use_proxy=True,
        browser=browser
    )
    global PROXY_LIST
    original_list = PROXY_LIST.copy()
    PROXY_LIST = [{"ip": ip, "port": port}]
    try:
        background_tasks.add_task(kroger_automation_task, test_request, browser)
        return {
            "message": f"Proxy test started for {ip}:{port} in {browser}",
            "check_status_at": f"/kroger/status?browser={browser}"
        }
    finally:
        def restore_proxies():
            time.sleep(2)
            global PROXY_LIST
            PROXY_LIST = original_list
        background_tasks.add_task(restore_proxies)

if __name__ == "__main__":
    try:
        ChromeDriverManager().install()
        GeckoDriverManager().install()
        logger.info("ChromeDriver and GeckoDriver installation successful")
    except Exception as e:
        logger.error(f"WebDriver installation failed: {str(e)}")
        logger.info("The API will still start, but automation may fail")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

