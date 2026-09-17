import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.banuri.edu.pk"
START_URL = f"{BASE_URL}/new-questions"

REQUEST_DELAY = 1

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "banuri_scraped.csv"
)

COLUMNS = [
    "fatwa_number",
    "category",
    "sub_category",
    "question",
    "answer",
    "url",
]


session = requests.Session()

session.headers.update({
    "User-Agent": "Islamic-Rag research scraper/1.0 (polite; contact project owner)",
})


def clean_text(text):
    """Extra spaces/newlines remove karta hai."""
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def get_page(url):
    """Website se page download karta hai."""
    try:
        print(f"GET: {url}")

        response = session.get(url, timeout=30)

        print(f"STATUS: {response.status_code}")

        if response.status_code != 200:
            print(f"WARNING: Could not fetch {url}")
            return None

        response.raise_for_status()
        return response.text

    except requests.RequestException as e:
        print(f"ERROR: {url}")
        print(e)
        return None

    finally:
        time.sleep(REQUEST_DELAY)


def get_question_links(html):
    """Listing page se individual question URLs nikalta hai."""

    soup = BeautifulSoup(html, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if not any(path in href for path in ("/fatwa/", "/fatawa/", "/readquestion/")):
            continue

        full_url = urljoin(BASE_URL, href)

        if full_url not in links:
            links.append(full_url)

    return links


def get_page_numbers(html):
    """Pagination se available page numbers nikalta hai."""

    soup = BeautifulSoup(html, "html.parser")

    page_numbers = []

    for a in soup.find_all("a", href=True):

        href = a["href"].strip()

        match = re.search(r"/new-questions/page/(\d+)", href)

        if match:
            page_numbers.append(int(match.group(1)))

    return sorted(set(page_numbers))


def extract_section(heading, stop_headings=()):
    """
    Collects text after a question/answer heading until the next section.
    """

    parts = []
    for element in heading.find_all_next():
        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if clean_text(element.get_text(" ", strip=True)) in stop_headings:
                break
        if element.name in ("p", "li"):
            text = clean_text(element.get_text(" ", strip=True))
            if text and text not in parts:
                parts.append(text)
    return " ".join(parts)


def find_heading(container, label):
    """Find a heading whose normalized text matches the Urdu section label."""
    for heading in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if clean_text(heading.get_text(" ", strip=True)) == label:
            return heading
    return None


def extract_detail(url, html):
    """Individual fatwa page se required fields extract karta hai."""

    soup = BeautifulSoup(html, "html.parser")

    box = soup.select_one(".sawal-jawab")

    if not box:
        print("WARNING: .sawal-jawab not found")
        return None

    # --------------------------------------------------
    # Fatwa Number
    # --------------------------------------------------

    fatwa_number = ""

    fatwa_element = box.select_one("#fatwa_number")

    if fatwa_element:
        fatwa_number = clean_text(
            fatwa_element.get_text(" ", strip=True)
        )

    if not fatwa_number:
        match = re.search(r"/(?:readquestion|fatwa|fatawa)/([0-9]+)", url)

        if match:
            fatwa_number = match.group(1)

    if not fatwa_number and soup.title:
        match = re.search(r"\b(\d{6,})\b", soup.title.get_text(" ", strip=True))
        if match:
            fatwa_number = match.group(1)

    # --------------------------------------------------
    # Question
    # --------------------------------------------------

    question = ""

    question_heading = find_heading(box, "سوال")

    if question_heading:
        question = extract_section(question_heading, ("جواب", "سوال پوچھیں"))

    # --------------------------------------------------
    # Answer
    # --------------------------------------------------

    answer = ""

    answer_heading = find_heading(box, "جواب")

    if answer_heading:
        answer = extract_section(answer_heading, ("سوال پوچھیں",))

    # --------------------------------------------------
    # Category + Sub Category
    # --------------------------------------------------

    categories = []

    for tag in box.select(".tag a"):
        text = clean_text(tag.get_text(" ", strip=True))
        if text and text not in categories:
            categories.append(text)

    category = categories[0] if len(categories) >= 1 else ""
    sub_category = categories[1] if len(categories) >= 2 else ""

    # --------------------------------------------------
    # Canonical URL
    # --------------------------------------------------

    canonical = soup.select_one(
        'link[rel="canonical"]'
    )

    if canonical and canonical.get("href"):
        canonical_url = urljoin(
            BASE_URL,
            canonical["href"].strip()
        )
    else:
        canonical_url = url

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if not question:
        print("WARNING: Question missing")

    if not answer:
        print("WARNING: Answer missing")

    return {
        "fatwa_number": fatwa_number,
        "category": category,
        "sub_category": sub_category,
        "question": question,
        "answer": answer,
        "url": canonical_url,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape Banuri Town fatawa into CSV.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit listing pages for testing; default crawls all discovered pages.",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=None,
        help="Limit detail pages for testing; default scrapes every discovered link.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("BANURI WEB SCRAPER")
    print("=" * 60)

    all_question_urls = []
    seen_urls = set()

    # --------------------------------------------------
    # Step 1: First page
    # --------------------------------------------------

    html = get_page(START_URL)

    if not html:
        print("Could not download first page.")
        return

    page_numbers = get_page_numbers(html)

    print(
        f"Pagination pages detected: "
        f"{page_numbers[:10]} ..."
    )

    # First page links
    links = get_question_links(html)
    seen_page_signatures = {tuple(links)}

    for url in links:
        if url not in seen_urls:
            seen_urls.add(url)
            all_question_urls.append(url)

    print(
        f"Page 1: found {len(links)} question links"
    )

    # --------------------------------------------------
    # Step 2: Remaining listing pages
    # --------------------------------------------------

    last_page = max(page_numbers, default=1)
    if args.max_pages:
        last_page = min(last_page, args.max_pages)

    print(f"Listing pages to scrape: 1-{last_page}")

    for page_number in range(2, last_page + 1):

        page_url = (
            f"{BASE_URL}/new-questions/page/"
            f"{page_number}"
        )

        html = get_page(page_url)

        if not html:
            print(
                f"Skipping page {page_number}"
            )
            continue

        links = get_question_links(html)

        page_signature = tuple(links)

        print(
            f"Page {page_number}: "
            f"found {len(links)} question links"
        )

        if not links:
            print(
                "No question links found. "
                "Stopping pagination."
            )
            break

        if page_signature in seen_page_signatures:
            print(
                "Duplicate listing page detected. "
                "Stopping pagination."
            )
            break

        seen_page_signatures.add(page_signature)

        for url in links:

            if url not in seen_urls:
                seen_urls.add(url)
                all_question_urls.append(url)

    # --------------------------------------------------
    # Step 3: Scrape individual fatwa pages
    # --------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"TOTAL UNIQUE QUESTION URLS: "
        f"{len(all_question_urls)}"
    )
    print("=" * 60)

    if args.max_details:
        all_question_urls = all_question_urls[:args.max_details]

    records = []

    for index, url in enumerate(
        all_question_urls,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(all_question_urls)}] "
            f"Scraping detail page..."
        )

        html = get_page(url)

        if not html:
            continue

        record = extract_detail(
            url,
            html
        )

        if record:

            # Only save useful records
            if record["question"] and record["answer"]:
                records.append(record)

            else:
                print(
                    "WARNING: Record incomplete, "
                    "not added."
                )

    # --------------------------------------------------
    # Step 4: Save CSV
    # --------------------------------------------------

    df = pd.DataFrame(
        records,
        columns=COLUMNS
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)

    print(
        f"Records saved: {len(df)}"
    )

    print(
        f"CSV file: {OUTPUT_FILE}"
    )

    print()
    print("Columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()