import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# יצירת תיקיית נתונים אישיים עם timestamp ייחודי
timestamp = str(int(time.time()))
user_data_dir = f"~/data_{timestamp}"

# הגדרת אפשרויות דפדפן עם תיקיית נתונים אישיים
chrome_options = Options()
chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
chrome_options.add_argument("--headless")  # במידה ורוצים להפעיל במצב חסר ממשק

# אתחול דפדפן עם דרייבר אוטומטי
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# פתיחת הקישור שלך
url = "https://www.shufersal.co.il/online/he/A43?shuf_source=shufersal_icon_dy&shuf_medium=iconS&shuf_campaign=independnet_day"
driver.get(url)

# שליפת ה-DIV על פי ה-ID שלו
category_div = driver.find_element_by_id("category")

# שליפת המאפיינים המותאמים אישית מתוך ה-DIV
category_id = category_div.get_attribute('data-category-id')
campaign = category_div.get_attribute('data-campaign')
medium = category_div.get_attribute('data-medium')

# הדפסת התוצאות
print("Category ID:", category_id)
print("Campaign:", campaign)
print("Medium:", medium)

# סגירת הדפדפן
driver.quit()

