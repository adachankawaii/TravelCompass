import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


CATEGORIES = {
    "attractions": {
        "label": "attractions",
        "seed_urls": [
            "https://www.tripadvisor.com/Attractions-g293925-Activities-Ho_Chi_Minh_City.html",
            "https://www.tripadvisor.com/Attractions-g293924-Activities-Hanoi.html",
            "https://www.tripadvisor.com/Attractions-g298085-Activities-Da_Nang.html",
            "https://www.tripadvisor.com/Attractions-g293928-Activities-Nha_Trang_Khanh_Hoa_Province.html",
            "https://www.tripadvisor.com/Attractions-g298082-Activities-Hoi_An_Quang_Nam_Province.html",
            "https://www.tripadvisor.com/Attractions-g293923-Activities-H_Long_Bay_Quang_Ninh_Province.html",
            "https://www.tripadvisor.com/Attractions-g469418-Activities-Phu_Quoc_Island_Kien_Giang_Province.html",
            "https://www.tripadvisor.com/Attractions-g293922-Activities-Da_Lat_Lam_Dong_Province.html",
            "https://www.tripadvisor.com/Attractions-g303942-Activities-Can_Tho_Mekong_Delta.html",
        ],
        "detail_href_contains": "/Attraction_Review-",
    },
    "restaurants": {
        "label": "restaurants",
        "seed_urls": [
            "https://www.tripadvisor.com/Restaurants-g293925-Ho_Chi_Minh_City.html",
            "https://www.tripadvisor.com/Restaurants-g293924-Hanoi.html",
            "https://www.tripadvisor.com/Restaurants-g298085-Da_Nang.html",
            "https://www.tripadvisor.com/Restaurants-g293928-Nha_Trang_Khanh_Hoa_Province.html",
            "https://www.tripadvisor.com/Restaurants-g298082-Hoi_An_Quang_Nam_Province.html",
            "https://www.tripadvisor.com/Restaurants-g293923-Halong_Bay_Quang_Ninh_Province.html",
            "https://www.tripadvisor.com/Restaurants-g469418-Phu_Quoc_Island_Kien_Giang_Province.html",
            "https://www.tripadvisor.com/Restaurants-g293922-Da_Lat_Lam_Dong_Province.html",
            "https://www.tripadvisor.com/Restaurants-g303942-Can_Tho_Mekong_Delta.html",
        ],
        "detail_href_contains": "/Restaurant_Review-",
    },
    "hotels": {
        "label": "hotels",
        "seed_urls": [
            "https://www.tripadvisor.com/Hotels-g293925-Ho_Chi_Minh_City-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g293924-Hanoi-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g298085-Da_Nang-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g293928-Nha_Trang_Khanh_Hoa_Province-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g298082-Hoi_An_Quang_Nam_Province-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g293923-Halong_Bay_Quang_Ninh_Province-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g469418-Phu_Quoc_Island_Kien_Giang_Province-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g293922-Da_Lat_Lam_Dong_Province-Hotels.html",
            "https://www.tripadvisor.com/Hotels-g303942-Can_Tho_Mekong_Delta-Hotels.html",
        ],
        "detail_href_contains": "/Hotel_Review-",
    },
}


def _clean_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def _parse_float(text: str) -> float | None:
    if not text:
        return None
    t = text.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    return float(m.group(1)) if m else None


def _parse_int(text: str) -> int | None:
    if not text:
        return None
    t = text.replace(".", "").replace(",", "")
    m = re.search(r"(\d+)", t)
    return int(m.group(1)) if m else None


def human_sleep(a=0.75, b=1.75):
    time.sleep(random.uniform(a, b))


def expect_any(page, selectors: list[str], timeout=8000):
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=timeout)
            return sel
        except PlaywrightTimeoutError:
            continue
    return None


def extract_list_item_urls(page, href_substring: str, max_items: int = 20) -> list[str]:
    """Extract URLs from listing page, limited to max_items"""
    anchors = []
    anchors += page.query_selector_all(f'a[href*="{href_substring}"]')
    anchors += page.query_selector_all('a[data-automation*="click-card"], a[data-test-target="view_detail"]')
    urls = []
    seen = set()
    
    for a in anchors:
        if len(urls) >= max_items:  # Dừng khi đã đủ số lượng
            break
            
        try:
            href = a.get_attribute("href")
        except Exception:
            continue
        if not href:
            continue
        if href_substring not in href:
            try:
                child = a.query_selector(f'a[href*="{href_substring}"]')
                if child:
                    href = child.get_attribute("href")
                else:
                    continue
            except Exception:
                continue
        abs_url = urljoin(page.url, href)
        p = urlparse(abs_url)
        key = f"{p.scheme}://{p.netloc}{p.path}"
        if key in seen:
            continue
        seen.add(key)
        urls.append(key)
    
    return urls[:max_items]  # Đảm bảo không vượt quá max_items


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((PlaywrightTimeoutError,)))

def extract_detail(page, url: str, category_label: str) -> dict:
    page.goto(url, wait_until="domcontentloaded", timeout=25000)
    human_sleep(0.8, 1.8)

    # Tên địa điểm
    name = None
    name_selectors = [
        'h1',
        'h1[data-test-target="top-info-header"]',
        'h1:visible',
    ]
    sel = expect_any(page, name_selectors, timeout=6000)
    if sel:
        try:
            name = _clean_text(page.query_selector(sel).inner_text())
        except Exception:
            name = None

    # Rating
    rating = None
    try:
        node = page.query_selector('span[aria-label$="of 5 bubbles"]')
        if node:
            rating = _parse_float(node.get_attribute('aria-label'))
    except Exception:
        pass
    if rating is None:
        try:
            node = page.query_selector('svg[aria-label]')
            if node:
                rating = _parse_float(node.get_attribute('aria-label'))
        except Exception:
            pass

    # Review count
    review_count = None
    review_candidates = []
    try:
        review_candidates = [
            _clean_text(el.inner_text()) for el in page.query_selector_all(
                'a[href*="#REVIEWS"]:visible, span:has-text("reviews"):visible, span:has-text("đánh giá"):visible, a:has-text("reviews"):visible, a:has-text("đánh giá"):visible'
            )
        ]
    except Exception:
        pass

    for txt in review_candidates:
        if re.search(r"\breviews?\b|đánh giá", txt, flags=re.I):
            n = _parse_int(txt)
            if n:
                review_count = n
                break

    city = None
    try:
        crumbs = page.query_selector_all('nav[aria-label="Breadcrumbs"] a')
        trail = [_clean_text(c.inner_text()) for c in crumbs if _clean_text(c.inner_text())]
        for i in range(len(trail)-1, -1, -1):
            t = trail[i]
            if not t:
                continue
            if any(key in t for key in ["Hotels", "Restaurants", "Things to Do", "Vacation", "Rentals", "Activities", "Khách sạn", "Nhà hàng", "Điểm tham quan"]):
                continue
            if t.lower() in ("vietnam", "asia", "châu á"):
                continue
            city = t
            break
    except Exception:
        city = None

    data = {
        "name": name,
        "city": city,
        "rating": rating,
        "review_count": review_count,
        "category": category_label,
        "scrape_date": datetime.now().date().isoformat(),
        "url": url,
    }
    return data


def accept_cookies_if_any(page):
    for sel in [
        '#onetrust-accept-btn-handler',
        'button#onetrust-accept-btn-handler',
        'button:has-text("I Accept")',
        'button:has-text("Accept all")',
        'button:has-text("Tôi đồng ý")',
    ]:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                human_sleep(0.5, 1.0)
                break
        except Exception:
            pass


def auto_scroll(page, steps=4, delta=800):
    """Giảm số lần scroll vì chỉ cần top 20"""
    for _ in range(steps):
        page.mouse.wheel(0, delta)
        human_sleep(0.3, 0.7)


def crawl_category(pw, category_key: str, items_per_city: int = 20, headless: bool = True, lang: str = "en-US,en;q=0.9,vi-VN,vi;q=0.8"):
    """
    Cào dữ liệu cho một category, chỉ lấy top items_per_city kết quả đầu tiên của mỗi thành phố
    """
    if category_key not in CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    conf = CATEGORIES[category_key]
    href_sub = conf["detail_href_contains"]
    seed_urls = conf.get("seed_urls", [])

    browser = pw.chromium.launch(headless=headless)
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        locale="vi-VN",
        viewport={"width": 1366, "height": 900},
        extra_http_headers={"Accept-Language": lang},
    )

    page = ctx.new_page()
    page.set_default_timeout(25000)

    all_items = []
    
    try:
        for i, start_url in enumerate(seed_urls):
            if not start_url:
                continue
                
            # Lấy tên thành phố từ URL để hiển thị progress
            city_name = start_url.split('-')[-1].split('.')[0].replace('_', ' ')
            print(f"Đang cào {category_key} tại {city_name} ({i+1}/{len(seed_urls)})...")
            
            page.goto(start_url, wait_until="domcontentloaded")
            human_sleep(1.0, 2.0)
            accept_cookies_if_any(page)

            # Chỉ scroll nhẹ để load top items
            auto_scroll(page, steps=3)
            
            # Lấy top items_per_city URLs từ trang đầu tiên
            list_urls = extract_list_item_urls(page, href_substring=href_sub, max_items=items_per_city)
            
            city_items = []
            for u in tqdm(list_urls, desc=f"  {city_name} - {category_key}"):
                try:
                    dpage = ctx.new_page()
                    d = extract_detail(dpage, u, category_label=category_key)
                    dpage.close()
                    if d.get("name"):
                        city_items.append(d)
                except Exception as e:
                    print(f"    Lỗi khi cào {u}: {e}")
                    pass
                finally:
                    human_sleep(0.5, 1.4)
            
            all_items.extend(city_items)
            print(f"  Đã cào {len(city_items)} items từ {city_name}")
            
    finally:
        ctx.close()
        browser.close()

    return all_items


def save_outputs(rows: list[dict], out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    # CSV
    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "city", "rating", "review_count", "category", "scrape_date", "url"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    # JSONL
    jsonl_path = f"{out_prefix}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def main():
    parser = argparse.ArgumentParser(description="TripAdvisor Vietnam scraper - Top 20 per city")
    parser.add_argument("--category", choices=["attractions", "restaurants", "hotels", "all"], default="all",
                        help="Danh mục cần cào")
    parser.add_argument("--items-per-city", type=int, default=20, 
                        help="Số items tối đa mỗi thành phố (mặc định: 20)")
    parser.add_argument("--out", type=str, default="outputs/tripadvisor_vietnam_top20",
                        help="Tiền tố tên file đầu ra (không kèm đuôi)")
    parser.add_argument("--headful", action="store_true", help="Mở trình duyệt hiển thị (không headless)")
    parser.add_argument("--lang", type=str, default="en-US,en;q=0.9,vi-VN,vi;q=0.8",
                        help="Header Accept-Language")

    args = parser.parse_args()

    print(f"Bắt đầu cào dữ liệu TripAdvisor Vietnam - Top {args.items_per_city} mỗi thành phố")
    
    with sync_playwright() as pw:
        categories = list(CATEGORIES.keys()) if args.category == "all" else [args.category]
        all_rows = []
        
        for cat in categories:
            print(f"\n=== Đang cào category: {cat.upper()} ===")
            rows = crawl_category(
                pw,
                category_key=cat,
                items_per_city=args.items_per_city,
                headless=not args.headful,
                lang=args.lang,
            )
            all_rows.extend(rows)
            print(f"Hoàn thành {cat}: {len(rows)} items")

    csv_path, jsonl_path = save_outputs(all_rows, args.out)
    print(f"\n=== HOÀN THÀNH ===")
    print(f"Tổng cộng: {len(all_rows)} items")
    print(f"Saved: {csv_path}")
    print(f"Saved: {jsonl_path}")


if __name__ == "__main__":
    main()