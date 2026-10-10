import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Root site URL
BASE_URL = "https://www.banuri.edu.pk"
OUTPUT_CSV = "../data/banuri_namaz_fatawa.csv"

def scrape_fatwa_detail(fatwa_url):
    """Extract Question, Answer, and Metadata from a detail page."""
    try:
        res = requests.get(fatwa_url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.content, "html.parser")
        
        # Get all text content
        full_text = soup.get_text(separator="\n").strip()
        
        # Check if invalid page
        if "404" in full_text or "صفحہ نہیں ملا" in full_text:
            return None

        # Look for Question (سوال) and Answer (جواب) keywords
        if "سوال" in full_text and "جواب" in full_text:
            parts = full_text.split("جواب", 1)
            q_part = parts[0].split("سوال")[-1].strip()
            a_part = parts[1].strip()
            
            # Clean up trailing site text if present
            a_part = a_part.split("دارالافتاء")[0].strip()
        else:
            return None

        fatwa_id = fatwa_url.split("/readquestion/")[1].split("/")[0] if "/readquestion/" in fatwa_url else str(int(time.time()))

        if q_part and a_part:
            return {
                "fatwa_number": fatwa_id,
                "question": q_part[:600],
                "answer": a_part,
                "category": "نماز و عبادات",
                "sub_category": "نماز",
                "url": fatwa_url
            }
    except Exception as e:
        print(f"Error reading {fatwa_url}: {e}")
    return None

def main():
    os.makedirs("../data", exist_ok=True)
    scraped_data = []

    print("Fetching fatwa links from Banuri homepage...")
    try:
        res = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            print(f"Failed to access {BASE_URL}. Status code: {res.status_code}")
            return

        soup = BeautifulSoup(res.content, "html.parser")
        
        # Extract all valid readquestion links
        fatwa_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/readquestion/" in href:
                if not href.startswith("http"):
                    href = f"{BASE_URL}{href}"
                fatwa_links.append(href)

        fatwa_links = list(set(fatwa_links))
        print(f"Found {len(fatwa_links)} fatwa links on Banuri.")

        for idx, link in enumerate(fatwa_links, 1):
            print(f"[{idx}/{len(fatwa_links)}] Scraping: {link}")
            item = scrape_fatwa_detail(link)
            if item:
                scraped_data.append(item)
                print(f" -> Successfully scraped Fatwa #{item['fatwa_number']}")
            time.sleep(1)

    except Exception as e:
        print(f"Error fetching main page: {e}")

    if scraped_data:
        df = pd.DataFrame(scraped_data)
        df.drop_duplicates(subset=["question"], inplace=True)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(df)} scraped fatwas into {OUTPUT_CSV}!")
    else:
        print("\nNo fatwas were extracted.")

if __name__ == "__main__":
    main()