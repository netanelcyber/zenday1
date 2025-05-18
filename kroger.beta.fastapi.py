import logging
import platform
import time
from typing import List, Optional

#import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

import asyncio

# Check the platform and set the event loop policy if on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    logging.info("Setting asyncio event loop policy to WindowsProactorEventLoopPolicy")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kroger_automation")

# FastAPI app
app = FastAPI(title="Kroger Automation API", version="1.0.0", description="API for automating Kroger website interactions.")

# Status dictionary
operation_status = {"status": "idle", "message": "", "timestamp": ""}


# Pydantic models
class Credentials(BaseModel):
    """
    Represents user credentials for logging into Kroger.
    """
    email: str = Field(..., description="Email address for the Kroger account.")
    password: str = Field(..., description="Password for the Kroger account.")


class KrogerRequest(BaseModel):
    """
    Represents a request to perform Kroger automation tasks.
    """
    credentials: Optional[Credentials] = Field(default_factory=lambda: Credentials(email="your-email@example.com", password="your-password"), description="User credentials for Kroger. If not provided, default values are used.")
    product: str = Field("milk", description="Name of the product to search for on Kroger.")
    headless: bool = Field(True, description="Whether to run the browser in headless mode (without a GUI).")


class ProductDetails(BaseModel):
    """
    Represents details of a product found on Kroger.
    """
    name: Optional[str] = Field(None, description="Name of the product.")
    price: Optional[str] = Field(None, description="Price of the product.")
    # Add other relevant details here


class SearchResult(BaseModel):
    """
    Represents a search result for a product on Kroger.
    """
    name: str = Field(..., description="Name of the product found in the search results.")


def update_status(status: str, message: str):
    """
    Updates the global operation status dictionary and logs the status change.

    Args:
        status (str): The new status of the operation (e.g., "starting", "success", "error").
        message (str): A descriptive message about the status.
    """
    operation_status.update({"status": status, "message": message, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    logger.info(f"[{status.upper()}] {message}")


# We don't need this function anymore since Playwright handles process cleanup properly
# But we'll keep it with a warning in case it's called anywhere
def kill_driver_process(pid: int):
    """
    This function is deprecated as Playwright handles its own process cleanup.

    Args:
        pid (int): The process ID of the browser process to kill.
    """
    logger.warning("kill_driver_process is deprecated with Playwright as it manages its own processes")


async def login_to_kroger(page, credentials):
    """
    Helper function to log into Kroger website.

    Args:
        page: Playwright page object
        credentials: User credentials

    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        update_status("navigating", "Opening Kroger login page...")

        # Try different approaches to access the site
        try:
            # First try: standard navigation with longer timeout
            await page.goto("https://www.kroger.com/signin", timeout=60000)
        except Exception as e:
            logger.warning(f"First attempt to load login page failed: {e}")
            try:
                # Second try: mobile user agent
                await page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"})
                await page.goto("https://www.kroger.com/signin", timeout=60000)
            except Exception as e2:
                logger.warning(f"Second attempt to load login page failed: {e2}")
                # Third try: Try the non-SSL version and let it redirect
                await page.goto("http://www.kroger.com", timeout=60000)
                # Navigate to signin from homepage
                try:
                    await page.click("a[href*='signin']", timeout=10000)
                except:
                    logger.warning("Could not find signin link, trying direct URL again")
                    await page.goto("https://www.kroger.com/signin", timeout=60000, wait_until="networkidle")

        # Wait for the login form
        try:
            await page.wait_for_selector("#email", timeout=30000)
        except Exception as e:
            logger.error(f"Login form not found: {e}")
            # Try to save screenshot for debugging
            try:
                await page.screenshot(path="login_error.png")
                logger.info("Saved error screenshot to login_error.png")
            except:
                pass
            return False

        update_status("logging_in", f"Logging in as {credentials.email}...")
        await page.fill("#email", credentials.email)
        await page.fill("#password", credentials.password)

        # Find and click the sign in button in multiple ways
        try:
            await page.click("//button[contains(text(), 'Sign In')]")
        except Exception as button_error:
            try:
                await page.click("button[type='submit']")
            except:
                logger.error(f"Could not click login button: {button_error}")
                return False

        try:
            # Check different possible outcomes after login
            success = False

            # Option 1: Wait for dashboard
            try:
                await page.wait_for_url("**/dashboard", timeout=30000)
                update_status("logged_in", "Successfully logged in!")
                success = True
            except PlaywrightTimeoutError:
                # Option 2: Check if we're on a different page but logged in
                try:
                    account_element = await page.query_selector("//a[contains(@href, 'account') or contains(text(), 'Account')]")
                    if account_element:
                        update_status("logged_in", "Successfully logged in (account element found)!")
                        success = True
                except:
                    pass

            if not success:
                # Option 3: Verification needed
                update_status("verification_needed", "Manual verification might be required. Waiting longer...")
                # Give more time for manual verification if needed
                await asyncio.sleep(60)

                if "/dashboard" in page.url or await page.query_selector("//a[contains(@href, 'account')]"):
                    update_status("logged_in", "Successfully logged in after waiting!")
                    return True
                else:
                    update_status("login_failed", "Login failed after extended wait")
                    return False

            return True
        except PlaywrightTimeoutError:
            return False
    except Exception as e:
        update_status("login_error", f"Error during login: {str(e)}")
        return False


async def kroger_automation_task(request: KrogerRequest):
    """
    Handles the Kroger automation task using Playwright.
    """
    browser = None
    browser_process_pid = None

    try:
        update_status("starting", "Setting up Playwright...")
        async with async_playwright() as p:
            browser_type = p.chromium
            browser_options = {
                "headless": request.headless,
                "args": [
                    "--disable-http2",  # Disable HTTP/2 protocol
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-web-security",  # Less strict security for automation
                    "--disable-features=IsolateOrigins,site-per-process"  # Helps with some sites
                ],
             #   "ignore_https_errors": True  # Ignore HTTPS errors
            }
            browser = await browser_type.launch(**browser_options)
            # In Playwright, we get the pid differently than in Selenium
            browser_process_pid = None  # We'll handle cleanup differently
            page = await browser.new_page()

            # Login to Kroger
            login_success = await login_to_kroger(page, request.credentials)
            if not login_success:
                return

            # Search for product
            update_status("searching", f"Searching for '{request.product}'...")
            await page.wait_for_selector("//input[@placeholder='Search']", timeout=25000)
            await page.fill("//input[@placeholder='Search']", request.product)
            await page.press("//input[@placeholder='Search']", "Enter")

            try:
                await page.wait_for_selector("div.ProductCard", timeout=25000)
                update_status("search_complete", "Search results loaded")
            except PlaywrightTimeoutError:
                update_status("search_no_results", f"No results found for '{request.product}'")
                return

            # Add to cart
            update_status("adding_to_cart", "Attempting to add first available product to cart...")
            add_buttons = await page.query_selector_all(
                "//button[contains(@aria-label, 'Add to Cart') or contains(text(), 'Add to Cart')]"
            )

            if add_buttons:
                added_to_cart = False
                for button in add_buttons:
                    if await button.is_visible():
                        try:
                            await button.click()
                            update_status("adding_to_cart", "Clicked 'Add to Cart' button.")
                            try:
                                await page.wait_for_selector("div.kds-Counter", timeout=10000)
                                update_status("success", f"'{request.product}' added to cart")
                                added_to_cart = True
                                break
                            except PlaywrightTimeoutError:
                                update_status("cart_error", "Item may not have been added to cart")
                        except Exception as e:
                            logger.warning(f"Could not click this Add to Cart button: {e}")
                            continue

                if not added_to_cart:
                    update_status("cart_error", "Could not add any product to cart")
            else:
                update_status("search_no_results", f"No 'Add to Cart' buttons found for '{request.product}'.")

    except PlaywrightTimeoutError as te:
        update_status("timeout_error", f"Timeout during operation: {te}")
    except Exception as e:
        update_status("error", f"An unexpected error occurred during the process: {str(e)}")
    finally:
        if browser:
            try:
                await browser.close()
                logger.info("Browser closed successfully.")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
                # Playwright handles its own process cleanup, no need to kill process


async def clear_kroger_cart_task(request: KrogerRequest):
    """
    Attempts to clear the Kroger shopping cart using Playwright.
    """
    browser = None
    browser_process_pid = None

    try:
        update_status("starting", "Setting up Playwright for clearing cart...")
        async with async_playwright() as p:
            browser_type = p.chromium
            browser_options = {
                "headless": request.headless,
            }
            browser = await browser_type.launch(**browser_options)
            browser_process_pid = browser.process.pid
            page = await browser.new_page()

            # Login to Kroger
            login_success = await login_to_kroger(page, request.credentials)
            if not login_success:
                return

            # Navigate to cart
            update_status("navigating_cart", "Navigating to the shopping cart...")
            await page.goto("https://www.kroger.com/cart", timeout=45000)

            # Check if cart is empty
            try:
                empty_cart = await page.query_selector("//div[contains(text(), 'Your cart is empty')]")
                if empty_cart and await empty_cart.is_visible():
                    update_status("cart_empty", "Cart is already empty.")
                    return
            except Exception:
                # Continue if we couldn't find the empty cart message
                pass

            # Try to find cart items
            try:
                await page.wait_for_selector("div.Cart-item", timeout=15000)
            except PlaywrightTimeoutError:
                update_status("cart_empty", "Cart appears to be empty.")
                return

            # Find and click remove buttons
            remove_buttons = await page.query_selector_all("//button[contains(@aria-label, 'Remove')]")
            num_items = len(remove_buttons)
            update_status("clearing_cart", f"Found {num_items} items to remove...")

            for i, button in enumerate(remove_buttons):
                try:
                    if await button.is_visible():
                        await button.click()
                        update_status("clearing_cart", f"Removed item {i+1} of {num_items}")
                        # Wait for the removal to process
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(f"Could not click remove button: {e}")

            # Verify cart is empty
            try:
                empty_cart = await page.query_selector("//div[contains(text(), 'Your cart is empty')]")
                if empty_cart and await empty_cart.is_visible():
                    update_status("cart_cleared", "Cart has been emptied successfully")
                else:
                    update_status("cart_partially_cleared", "Some items may remain in the cart")
            except Exception:
                update_status("cart_cleared", f"Attempted to remove {num_items} items from the cart")

    except PlaywrightTimeoutError as te:
        update_status("timeout_error", f"Timeout during clear cart operation: {te}")
    except Exception as e:
        update_status("error", f"An unexpected error occurred while clearing the cart: {e}")
    finally:
        if browser:
            try:
                await browser.close()
                logger.info("Browser closed after clear cart.")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
                if browser_process_pid:
                    kill_driver_process(browser_process_pid)


async def get_product_details_task(product_name: str, headless: bool = True):
    """
    Helper function to fetch product details.
    """
    browser = None

    try:
        async with async_playwright() as p:
            browser_type = p.chromium
            browser_options = {
                "headless": headless,
                "args": [
                    "--disable-http2",  # Disable HTTP/2 protocol
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-web-security",  # Less strict security for automation
                    "--disable-features=IsolateOrigins,site-per-process"  # Helps with some sites
                ],
             #   "ignore_https_errors": True  # Ignore HTTPS errors
            }
            browser = await browser_type.launch(**browser_options)
            page = await browser.new_page()

            # Navigate to search results
            try:
                await page.goto(
                    f"https://www.kroger.com/search?query={product_name.replace(' ', '%20')}&searchType=natural",
                    timeout=45000,
                )
            except Exception as e:
                logger.warning(f"Error navigating to search page: {e}")
                # Try an alternative approach
                await page.goto("https://www.kroger.com", timeout=45000)
                await page.fill("//input[@placeholder='Search']", product_name)
                await page.press("//input[@placeholder='Search']", "Enter")


            try:
                await page.wait_for_selector("div.ProductCard", timeout=15000)
            except PlaywrightTimeoutError:
                return None, "No products found"

            # Get first product details
            first_product = await page.query_selector("div.ProductCard")
            if not first_product:
                return None, "No product card found"

            details = ProductDetails()

            # Get product name
            title_element = await first_product.query_selector("h3.ProductCard-title")
            if title_element:
                details.name = await title_element.inner_text()

            # Get product price
            price_element = await first_product.query_selector("div.ProductCard-priceInner")
            if price_element:
                details.price = await price_element.inner_text()

            return details, None

    except Exception as e:
        return None, str(e)
    finally:
        if browser:
            await browser.close()



@app.post("/kroger/add-to-cart", tags=["Kroger"])
async def add_to_cart(request: KrogerRequest, background_tasks: BackgroundTasks):
    """
    Initiates the process of adding a product to the Kroger cart.
    """
    # List of statuses that indicate no task is currently running
    idle_statuses = [
        "idle", "completed", "error", "success", "system_error", "cart_error",
        "timeout_error", "element_not_found_error", "search_no_results",
        "verification_needed"
    ]

    if operation_status["status"] not in idle_statuses:
        return JSONResponse(
            status_code=409,
            content={"message": "Another automation task is already running", "current_status": operation_status},
        )

    update_status("queued", "Task queued")
    background_tasks.add_task(kroger_automation_task, request)
    return {"message": "Kroger automation started", "status": "queued", "check_status_at": "/kroger/status"}


@app.get("/kroger/status", tags=["Kroger"])
async def get_status():
    """
    Retrieves the current status of the Kroger automation process.
    """
    return operation_status


@app.post("/kroger/reset", tags=["Kroger"])
async def reset_status():
    """
    Resets the Kroger automation process status to "idle".
    """
    update_status("idle", "Status manually reset")
    return {"message": "Status reset successful"}


@app.get("/kroger/product/details", tags=["Kroger"])
async def get_product_details(
    product_name: str = Query(..., description="Name of the product to search for"),
    headless: bool = True
):
    """
    Searches for a product on Kroger and returns details of the first result.
    """
    update_status("searching_details", f"Searching for details of '{product_name}'...")

    details, error = await get_product_details_task(product_name, headless)

    if error:
        if "No products found" in error or "No product card found" in error:
            update_status("not_found", f"No product found with the name '{product_name}'.")
            raise HTTPException(status_code=404, detail=f"Product '{product_name}' not found")
        elif "Timeout" in error:
            update_status("timeout_error", f"Timeout while searching for details: {error}")
            raise HTTPException(status_code=408, detail="Timeout during product details search")
        else:
            update_status("error", f"Error fetching product details: {error}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {error}")

    update_status("details_found", f"Details found for '{product_name}'.")
    return details



@app.get("/kroger/search", tags=["Kroger"], response_model=List[SearchResult])
async def get_search_results(query: str = Query(..., description="Search term"), headless: bool = True):
    """
    Searches Kroger for a product and returns a list of the search results.
    """
    browser = None

    try:
        update_status("searching_results", f"Searching for '{query}'...")
        async with async_playwright() as p:
            browser_type = p.chromium
            browser_options = {
                "headless": headless,
            }
            browser = await browser_type.launch(**browser_options)
            page = await browser.new_page()

            # Navigate to search results page
            await page.goto(
                f"https://www.kroger.com/search?query={query.replace(' ', '%20')}&searchType=natural",
                timeout=30000
            )

            try:
                await page.wait_for_selector("div.ProductCard", timeout=15000)
            except PlaywrightTimeoutError:
                update_status("search_no_results", f"No results found for '{query}'")
                return []

            # Get all product cards
            product_cards = await page.query_selector_all("div.ProductCard")
            results = []

            for card in product_cards:
                title_element = await card.query_selector("h3.ProductCard-title")
                if title_element:
                    product_name = await title_element.inner_text()
                    results.append(SearchResult(name=product_name))

            update_status("results_found", f"{len(results)} results found for '{query}'.")
            return results

    except PlaywrightTimeoutError as te:
        update_status("timeout_error", f"Timeout during search: {te}")
        raise HTTPException(status_code=408, detail="Timeout during search")
    except Exception as e:
        update_status("error", f"Error fetching search results: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if browser:
            await browser.close()



@app.post("/kroger/cart/clear", tags=["Kroger"])
async def clear_cart(request: KrogerRequest, background_tasks: BackgroundTasks):
    """
    Initiates the process of clearing the Kroger shopping cart.
    """
    # List of statuses that indicate no task is currently running
    idle_statuses = [
        "idle", "completed", "error", "success", "system_error", "cart_error",
        "timeout_error", "element_not_found_error", "search_no_results",
        "verification_needed", "cart_cleared", "cart_empty"
    ]

    if operation_status["status"] not in idle_statuses:
        return JSONResponse(
            status_code=409,
            content={"message": "Another automation task is already running", "current_status": operation_status},
        )

    update_status("queued", "Clear cart task queued")
    background_tasks.add_task(clear_kroger_cart_task, request)
    return {"message": "Kroger clear cart automation started", "status": "queued", "check_status_at": "/kroger/status"}


@app.get("/", tags=["Root"])
async def root():
    """
    Provides basic information about the API.
    """
    return {
        "api": "Kroger Automation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /kroger/add-to-cart": "Start automation",
            "GET /kroger/status": "Check automation status",
            "POST /kroger/reset": "Reset status",
            "GET /kroger/product/details": "Get product details",
            "GET /kroger/search": "Get search results",
            "POST /kroger/cart/clear": "Clear Cart"
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    """
    Checks the health of the API.
    """
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


if __name__ == "__main__":
    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = ["localhost:8050"]
    print("Starting server on http://localhost:8050")
    print("API documentation available at http://localhost:8050/docs")
    asyncio.run(serve(app, config))

