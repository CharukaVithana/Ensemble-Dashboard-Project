"""POLITE scraper TEMPLATE (UNTESTED - the site layouts change, so you MUST adjust the selectors).
Before running: read the site's Terms of Use and robots.txt. Keep it small (a few hundred ads), add delays.
Alternative that is always allowed: copy ads by hand into data/vehicles_sri_lanka.csv with the columns below.
Output columns: make, model, year, mileage, engine, fuel, transmission, district, price
"""
import csv, time, urllib.robotparser
import requests
from bs4 import BeautifulSoup

BASE = "https://riyasewana.com"                    # check the real listing URL pattern in your browser
LIST_URL = BASE + "/search/cars?page={page}"       # <-- EDIT
HEADERS = {"User-Agent": "UniversityAssignmentBot/1.0 (student project; contact: your_email@example.com)"}
DELAY_S, MAX_PAGES = 3, 10

rp = urllib.robotparser.RobotFileParser(BASE + "/robots.txt"); rp.read()


def get(url):
    if not rp.can_fetch(HEADERS["User-Agent"], url):
        print("Blocked by robots.txt, skipping:", url); return None
    time.sleep(DELAY_S)
    r = requests.get(url, headers=HEADERS, timeout=20)
    return BeautifulSoup(r.text, "html.parser") if r.ok else None


def parse_ad(url):
    """Return a dict for one ad. EDIT the selectors after inspecting the ad page (right-click > Inspect)."""
    soup = get(url)
    if soup is None:
        return None
    # Example idea: ad details are often in a table of label/value pairs.
    details = {}
    for row in soup.select("table tr"):
        cells = [c.get_text(strip=True) for c in row.select("td")]
        if len(cells) >= 2:
            details[cells[0].lower()] = cells[1]
    return {"make": details.get("make"), "model": details.get("model"), "year": details.get("yom"),
            "mileage": details.get("mileage (km)"), "engine": details.get("engine (cc)"),
            "fuel": details.get("fuel type"), "transmission": details.get("transmission"),
            "district": details.get("district"), "price": details.get("price")}


if __name__ == "__main__":
    rows = []
    for page in range(1, MAX_PAGES + 1):
        soup = get(LIST_URL.format(page=page))
        if soup is None:
            break
        links = [a["href"] for a in soup.select("li.item h2 a")]       # <-- EDIT selector for ad links
        for href in links:
            ad = parse_ad(href if href.startswith("http") else BASE + href)
            if ad and ad["price"]:
                rows.append(ad)
        print(f"page {page}: total ads so far {len(rows)}")
    with open("data/vehicles_sri_lanka.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["make", "model", "year", "mileage", "engine", "fuel", "transmission", "district", "price"])
        w.writeheader(); w.writerows(rows)
    print("Saved", len(rows), "rows. Note the collection date in your report.")
