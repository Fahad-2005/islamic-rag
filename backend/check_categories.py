import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BASE_URL = "https://www.banuri.edu.pk"

def inspect_site():
    print("جامعہ بنوری کی ویب سائٹ سے تمام مرکزی کیٹیگریز اور لنکس تلاش کیے جا رہے ہیں...\n")
    
    try:
        res = requests.get(BASE_URL, headers=HEADERS, timeout=15)
        if res.status_code != 200:
            print(f"ویب سائٹ تک رسائی ناکام رہی۔ اسٹیٹس کوڈ: {res.status_code}")
            return

        soup = BeautifulSoup(res.content, "html.parser")
        
        all_links = soup.find_all("a", href=True)
        categories = {}

        # عبادات اور 5 ارکان سے متعلق الفاظ کی فہرست
        target_keywords = ["نماز", "طہارت", "زکوۃ", "زکوٰۃ", "روزه", "روزہ", "حج", "عبادات", "وضو", "غسل", "فتاوی", "فتوی"]

        for a in all_links:
            text = a.get_text().strip()
            href = a["href"]
            
            if not href.startswith("http"):
                href = f"{BASE_URL}{href}"

            # اگر متن میں عبادات کا کوئی لفظ موجود ہو
            if any(keyword in text for keyword in target_keywords):
                categories[text] = href
            elif any(pattern in href for pattern in ["/cat/", "/category/", "/readquestion/"]):
                if text and len(text) > 2:
                    categories[text] = href

        if categories:
            print(f"کل {len(categories)} اہم کیٹیگریز/صفحات مل گئے ہیں:\n")
            for title, link in categories.items():
                print(f"📌 کیٹیگری: {title}")
                print(f"   لینک:    {link}")
                print("-" * 60)
        else:
            print("کوئی مخصوص کیٹیگری لنک نہیں ملا۔ ویب سائٹ پر موجود تمام اہم لنکس:")
            for a in all_links[:25]:
                href = a["href"]
                text = a.get_text().strip()
                if text:
                    print(f"  - {text}: {href}")

    except Exception as e:
        print(f"خرابی: {e}")

if __name__ == "__main__":
    inspect_site()