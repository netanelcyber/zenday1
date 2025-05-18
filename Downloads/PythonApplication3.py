from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import uvicorn
from pydantic import BaseModel
from typing import Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kroger-automation")

# Create FastAPI app
app = FastAPI(title="Kroger Automation API", 
              description="API to login to Kroger and add products to cart",
              version="1.0.0")

# Define request models
class Credentials(BaseModel):
    email: str = "nsh531@gmail.com"
    password: str = "318962420A"
    
class KrogerRequest(BaseModel):
    credentials: Optional[Credentials] = Credentials()
    product: str = "milk"
    headless: bool = False

# Global variable to store operation status
operation_status = {"status": "idle", "message": "", "timestamp": ""}

def update_status(status: str, message: str):
    """Update the global operation status"""
    operation_status["status"] = status
    operation_status["message"] = message
    operation_status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Status updated: {status} - {message}")

def kroger_automation_task(request: KrogerRequest):
    """Background task to automate Kroger login and add to cart"""
    try:
        update_status("starting", "Setting up webdriver...")
        
        # Set up the webdriver (Chrome in this example)
        options = webdriver.ChromeOptions()
        if request.headless:
            options.add_argument('--headless')
        
        # Additional options to make Chrome more stable in server environments
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            # Step 1: Navigate to Kroger's login page
            update_status("navigating", "Navigating to Kroger's website...")
            driver.get("https://www.kroger.com/signin")
            
            # Wait for the page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "email"))
            )
            
            # Step 2: Enter login credentials
            update_status("logging_in", f"Logging in with email {request.credentials.email}...")
            driver.find_element(By.ID, "email").send_keys(request.credentials.email)
            driver.find_element(By.ID, "password").send_keys(request.credentials.password)
            
            # Click the sign-in button
            driver.find_element(By.XPATH, "//button[contains(text(), 'Sign In')]").click()
            
            # Wait for login to complete and dashboard to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.url_contains("/dashboard")
                )
                update_status("logged_in", "Successfully logged in!")
            except TimeoutException:
                update_status("verification_needed", "Login may require additional verification.")
                # Wait longer for manual verification if needed
                time.sleep(30)
            
            # Step 3: Search for a product
            update_status("searching", f"Searching for {request.product}...")
            
            # Navigate to search page and enter product
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search']"))
            )
            search_box.clear()
            search_box.send_keys(request.product)
            search_box.submit()
            
            # Wait for search results
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.ProductCard"))
            )
            update_status("search_complete", "Search results loaded.")
            
            # Step 4: Add the first product to cart
            try:
                # Find and click the first "Add to Cart" button
                update_status("adding_to_cart", "Attempting to add product to cart...")
                add_to_cart_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Add to Cart') or contains(text(), 'Add to Cart')]"))
                )
                add_to_cart_button.click()
                
                # Wait for the item to be added
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.kds-Counter"))
                )
                update_status("success", f"Successfully added {request.product} to cart!")
                
            except Exception as e:
                update_status("cart_error", f"Error adding product to cart: {str(e)}")
        
        except Exception as e:
            update_status("error", f"Process error: {str(e)}")
        
        finally:
            # Always close the browser when done
            driver.quit()
            if operation_status["status"] not in ["success", "cart_error", "error"]:
                update_status("completed", "Process completed but with unknown result")
    
    except Exception as e:
        update_status("system_error", f"System error: {str(e)}")

@app.post("/kroger/add-to-cart", tags=["Kroger"])
async def add_to_cart(request: KrogerRequest, background_tasks: BackgroundTasks):
    """
    Start a Kroger login and add-to-cart automation process.
    
    This endpoint triggers a background task that will:
    1. Log into Kroger with the provided credentials
    2. Search for the specified product
    3. Add the first matching product to the cart
    
    The process runs asynchronously, and you can check its status with the /status endpoint.
    """
    # Reset operation status
    update_status("queued", "Task queued and ready to start")
    
    # Add the task to the background tasks
    background_tasks.add_task(kroger_automation_task, request)
    
    return {"message": "Kroger automation process started", 
            "status": "queued",
            "check_status_at": "/kroger/status"}

@app.get("/kroger/status", tags=["Kroger"])
async def get_status():
    """Get the current status of the Kroger automation process"""
    return operation_status

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "api": "Kroger Automation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /kroger/add-to-cart": "Start the Kroger login and add-to-cart process",
            "GET /kroger/status": "Check the status of the current operation"
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8841, reload=True)