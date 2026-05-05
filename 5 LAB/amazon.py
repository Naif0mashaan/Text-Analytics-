"""
Name: Alhathal Naif Mashaan D
Student ID: SW01084553


"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv


# SETUP DRIVER


chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--lang=en-US")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 20)

# Remove Selenium automation flag
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")


# AMAZON REVIEW LINK


base_url = "https://www.amazon.com/product-reviews/0061122416/?reviewerType=all_reviews"

all_reviews = []


# SCRAPE 5 PAGES


for page in range(1, 6):
    print(f"Scraping page {page}...")
    
    driver.get(base_url + f"&pageNumber={page}")
    
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-hook='review']")))
    except:
        print("Reviews not found. Possible CAPTCHA or block.")
        break

    reviews = driver.find_elements(By.XPATH, "//div[@data-hook='review']")

    print(f"Found {len(reviews)} reviews on this page.")

    for review in reviews:
        try:
            name = review.find_element(By.CLASS_NAME, "a-profile-name").text
        except:
            name = "N/A"

        try:
            date = review.find_element(By.XPATH, ".//span[@data-hook='review-date']").text
        except:
            date = "N/A"

        try:
            content = review.find_element(By.XPATH, ".//span[@data-hook='review-body']").text
        except:
            content = "N/A"

        all_reviews.append([name, date, content])

    time.sleep(4)


# SAVE TO CSV


with open("amazon_reviews.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Reviewer Name", "Review Date", "Review Content"])
    writer.writerows(all_reviews)

print("Finished. Total reviews scraped:", len(all_reviews))

driver.quit()