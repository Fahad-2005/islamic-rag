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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = DATA_DIR / "banuri_scraped.csv"
URLS_FILE = DATA_DIR / "banuri_question_urls.csv"

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
    "User-Agent": (
        "Islamic-Rag research scraper/1.0 "
        "(polite; contact project owner)"
    ),
})


def clean_text(text):
    
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def get_page(url):
   
    try:
        print(f"GET: {url}")

        response = session.get(url, timeout=30)

        print(f"STATUS: {response.status_code}")

        if response.status_code != 200:
            print(f"WARNING: Could not fetch {url}")
            return None

        response.raise_for_status()
        return response.text

    except requests.RequestException as error:
        print(f"ERROR while fetching: {url}")
        print(error)
        return None

    finally:
        time.sleep(REQUEST_DELAY)


def get_question_links(html):
    
    soup = BeautifulSoup(html, "html.parser")

    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        if not any(
            path in href
            for path in ("/fatwa/", "/fatawa/", "/readquestion/")
        ):
            continue

        full_url = urljoin(BASE_URL, href)

        if full_url not in links:
            links.append(full_url)

    return links


def get_page_numbers(html):
   
    soup = BeautifulSoup(html, "html.parser")

    page_numbers = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        match = re.search(
            r"/new-questions/page/(\d+)",
            href
        )

        if match:
            page_numbers.append(int(match.group(1)))

    return sorted(set(page_numbers))


def extract_section(heading, stop_headings=()):
   
    parts = []

    for element in heading.find_all_next():
        if element.name in (
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ):
            heading_text = clean_text(
                element.get_text(" ", strip=True)
            )

            if heading_text in stop_headings:
                break

        if element.name in ("p", "li"):
            text = clean_text(
                element.get_text(" ", strip=True)
            )

            if text and text not in parts:
                parts.append(text)

    return " ".join(parts)


def find_heading(container, label):
    
    for heading in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        heading_text = clean_text(
            heading.get_text(" ", strip=True)
        )

        if heading_text == label:
            return heading

    return None


def extract_detail(url, html):
   
    soup = BeautifulSoup(html, "html.parser")

    box = soup.select_one(".sawal-jawab")

    if not box:
        print("WARNING: .sawal-jawab not found")
        return None

    # Fatwa number
    fatwa_number = ""

    fatwa_element = box.select_one("#fatwa_number")

    if fatwa_element:
        fatwa_number = clean_text(
            fatwa_element.get_text(" ", strip=True)
        )

    if not fatwa_number:
        match = re.search(
            r"/(?:readquestion|fatwa|fatawa)/([0-9]+)",
            url
        )

        if match:
            fatwa_number = match.group(1)

    if not fatwa_number and soup.title:
        match = re.search(
            r"\b(\d{6,})\b",
            soup.title.get_text(" ", strip=True)
        )

        if match:
            fatwa_number = match.group(1)

    # Question
    question = ""

    question_heading = find_heading(box, "سوال")

    if question_heading:
        question = extract_section(
            question_heading,
            ("جواب", "سوال پوچھیں")
        )

    # Answer
    answer = ""

    answer_heading = find_heading(box, "جواب")

    if answer_heading:
        answer = extract_section(
            answer_heading,
            ("سوال پوچھیں",)
        )

    # Categories
    categories = []

    for tag in box.select(".tag a"):
        text = clean_text(
            tag.get_text(" ", strip=True)
        )

        if text and text not in categories:
            categories.append(text)

    category = categories[0] if len(categories) >= 1 else ""
    sub_category = categories[1] if len(categories) >= 2 else ""

    # Canonical URL
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


def load_existing_records():
   
    if not OUTPUT_FILE.exists():
        return []

    try:
        df = pd.read_csv(
            OUTPUT_FILE,
            encoding="utf-8-sig"
        )

        df = df.fillna("")

        records = df.to_dict("records")

        print(
            f"Existing records loaded: {len(records)}"
        )

        return records

    except Exception as error:
        print(
            f"WARNING: Could not load existing CSV: {error}"
        )

        return []


def save_records(records):
    
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        records,
        columns=COLUMNS
    )

    if not df.empty:
        df = df.drop_duplicates(
            subset=["url"],
            keep="last"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Checkpoint saved: {len(df)} records"
    )


def save_urls(urls):
   
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame({
        "url": sorted(set(urls))
    })

    df.to_csv(
        URLS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"URL list saved: {len(df)} URLs"
    )


def load_saved_urls():
    
    if not URLS_FILE.exists():
        return []

    try:
        df = pd.read_csv(
            URLS_FILE,
            encoding="utf-8-sig"
        )

        return df["url"].dropna().astype(str).tolist()

    except Exception as error:
        print(
            f"WARNING: Could not load URL list: {error}"
        )

        return []


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Banuri Town fatawa into CSV "
            "with checkpoint and resume support."
        )
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Limit listing pages for testing. "
            "Default: all discovered pages."
        ),
    )

    parser.add_argument(
        "--max-details",
        type=int,
        default=None,
        help=(
            "Limit detail pages for testing. "
            "Default: all discovered URLs."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=10,
        help=(
            "Save after this many new records. "
            "Default: 10."
        ),
    )

    parser.add_argument(
        "--refresh-urls",
        action="store_true",
        help=(
            "Recrawl listing pages instead of loading the saved URL list."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("BANURI WEB SCRAPER - RESUMABLE VERSION")
    print("=" * 60)

    records = load_existing_records()

    existing_urls = {
        str(record.get("url", "")).strip()
        for record in records
        if record.get("url")
    }

    existing_fatwa_numbers = {
        str(record.get("fatwa_number", "")).strip()
        for record in records
        if record.get("fatwa_number")
    }

    # Load previously discovered URLs if available
    all_question_urls = (
        []
        if args.refresh_urls
        else load_saved_urls()
    )
    seen_urls = set(all_question_urls)

    # If URLs have not been discovered yet, crawl listing pages
    if not all_question_urls:
        print("Discovering question URLs...")

        html = get_page(START_URL)

        if not html:
            print("Could not download first page.")
            return

        page_numbers = get_page_numbers(html)

        print(
            f"Pagination pages detected: "
            f"{page_numbers[:10]} ..."
        )

        first_links = get_question_links(html)

        for url in first_links:
            if url not in seen_urls:
                seen_urls.add(url)
                all_question_urls.append(url)

        print(
            f"Page 1: found {len(first_links)} question links"
        )

        last_page = max(page_numbers, default=1)

        if args.max_pages:
            last_page = min(
                last_page,
                args.max_pages
            )

        print(
            f"Listing pages to scrape: 1-{last_page}"
        )

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

            new_links = 0

            for url in links:
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_question_urls.append(url)
                    new_links += 1

            print(
                f"New URLs added: {new_links}"
            )

            # Save URL progress after every listing page
            save_urls(all_question_urls)

    else:
        print(
            f"Loaded saved URL list: "
            f"{len(all_question_urls)} URLs"
        )

    if args.max_details == 0:
        save_urls(all_question_urls)
        print(
            "Detail scraping skipped because --max-details is 0."
        )
        return

    print()
    print("=" * 60)
    print(
        f"TOTAL UNIQUE QUESTION URLS: "
        f"{len(all_question_urls)}"
    )
    print("=" * 60)

    urls_to_scrape = [
        url
        for url in all_question_urls
        if url not in existing_urls
    ]

    if args.max_details:
        urls_to_scrape = urls_to_scrape[
            :args.max_details
        ]

    print(
        f"URLs remaining to scrape: "
        f"{len(urls_to_scrape)}"
    )

    new_records_since_checkpoint = 0

    try:
        for index, url in enumerate(
            urls_to_scrape,
            start=1
        ):
            print()
            print(
                f"[{index}/{len(urls_to_scrape)}] "
                f"Scraping detail page..."
            )

            html = get_page(url)

            if not html:
                continue

            record = extract_detail(
                url,
                html
            )

            if not record:
                continue

            question = record["question"]
            answer = record["answer"]
            fatwa_number = str(
                record["fatwa_number"]
            ).strip()

            if not question or not answer:
                print(
                    "WARNING: Incomplete record. "
                    "Not added."
                )
                continue

            if record["url"] in existing_urls:
                print(
                    "Already saved by URL. Skipping."
                )
                continue

            if (
                fatwa_number
                and fatwa_number in existing_fatwa_numbers
            ):
                print(
                    "Duplicate fatwa number. Skipping."
                )
                continue

            records.append(record)

            existing_urls.add(record["url"])

            if fatwa_number:
                existing_fatwa_numbers.add(
                    fatwa_number
                )

            new_records_since_checkpoint += 1

            print(
                f"Record added. "
                f"Total records: {len(records)}"
            )

            if (
                new_records_since_checkpoint
                >= args.checkpoint
            ):
                save_records(records)
                new_records_since_checkpoint = 0

    except KeyboardInterrupt:
        print()
        print(
            "Interrupted by user. "
            "Saving current progress..."
        )

    finally:
        save_records(records)
        save_urls(all_question_urls)

    print()
    print("=" * 60)
    print("SCRAPING SESSION FINISHED")
    print("=" * 60)
    print(
        f"Total records saved: {len(records)}"
    )
    print(
        f"CSV file: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()