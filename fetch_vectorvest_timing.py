#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path

import pandas as pd
import requests
import xml.etree.ElementTree as ET


def load_cookies_into_context(ctx, cookies_path: str):
    """Load cookies from a JSON file and inject them into the Playwright context."""
    path = Path(cookies_path)
    if not path.exists():
        raise FileNotFoundError(f"Cookie file not found: {cookies_path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cookie file is not valid JSON: {cookies_path}") from exc

    if isinstance(raw, dict) and "cookies" in raw:
        cookies = raw["cookies"]
    elif isinstance(raw, list):
        cookies = raw
    else:
        raise ValueError(
            "Cookie JSON must be a list of cookie objects or a dict containing a 'cookies' list"
        )

    def _normalize_same_site(value):
        if not value:
            return None
        v = str(value).strip().lower()
        if v in {"lax", "strict", "none"}:
            return v.capitalize()
        if v in {"no_restriction", "unspecified"}:
            return None
        return v.capitalize()

    formatted = []
    for idx, cookie in enumerate(cookies, start=1):
        name = cookie.get("name")
        if not name:
            print(f"Skipping cookie #{idx}: missing name")
            continue

        value = cookie.get("value", "")
        domain = cookie.get("domain")
        url = cookie.get("url")

        if domain:
            domain = domain.lstrip()
            domain = domain.lstrip('.')
        elif url:
            domain = None
        else:
            print(f"Skipping cookie '{name}': missing domain/url")
            continue

        entry = {
            "name": name,
            "value": value,
            "path": cookie.get("path") or "/",
            "secure": bool(cookie.get("secure")),
            "httpOnly": bool(cookie.get("httpOnly")),
        }

        same_site = _normalize_same_site(cookie.get("sameSite"))
        if same_site:
            entry["sameSite"] = same_site

        expires = cookie.get("expirationDate") or cookie.get("expires")
        if expires:
            entry["expires"] = float(expires)

        if domain:
            entry["domain"] = domain
        elif url:
            entry["url"] = url

        formatted.append(entry)

    if not formatted:
        print(f"No usable cookies loaded from {cookies_path}")
        return

    ctx.add_cookies(formatted)
    print(f"Loaded {len(formatted)} cookies from {cookies_path}")


def get_latest_link_from_feed(feed_url: str) -> str:
    """Return the newest post link from an RSS feed."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; VVFetcher/1.0)"}
    resp = requests.get(feed_url, headers=headers, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    channel = root.find('channel')
    if channel is None:
        raise RuntimeError(f"No <channel> element found in feed: {feed_url}")
    item = channel.find('item')
    if item is None:
        raise RuntimeError(f"Feed has no <item> entries: {feed_url}")
    link = item.find('link')
    if link is None or not (link.text and link.text.strip()):
        raise RuntimeError(f"Feed item missing <link>: {feed_url}")
    latest_link = link.text.strip()
    print(f"Latest link from feed: {latest_link}")
    return latest_link

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def extract_table_by_name(page, target_name="timing", timeout_ms=30000):
    """
    Heuristic:
    - Look for a table that is preceded by a heading containing 'Timing'
    - OR a table that contains a caption containing 'Timing'
    - OR a table whose first header row contains 'Timing' (rare)
    Returns: list[dict] rows
    """
    target = _normalize(target_name)

    # 1) Table with <caption>Timing</caption>
    tables = page.locator("table")
    n = tables.count()
    for i in range(n):
        t = tables.nth(i)
        try:
            cap = t.locator("caption").first
            if cap.count() and target in _normalize(cap.inner_text()):
                return table_to_records(t)
        except PWTimeout:
            pass

    # 2) Heading containing Timing, then next table after it
    # Common newsletter markup: h1/h2/h3 or strong label then table
    headings = page.locator("h1, h2, h3, h4, h5, h6, strong, b")
    hn = headings.count()
    for i in range(hn):
        h = headings.nth(i)
        txt = _normalize(h.inner_text())
        if target in txt:
            # get the next table in DOM after this heading
            # xpath: following::table[1]
            candidate = page.locator(f"xpath=({h.element_handle().evaluate('el => el.tagName')})")  # noop
            next_table = page.locator("xpath=following::table[1]", has=h)  # may not work reliably

    # More reliable: use an xpath anchored to the heading element itself
    # Find *any* element whose text includes Timing, then take following table[1]
    timing_anchor = page.locator(
        "xpath=//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::strong or self::b]"
        f"[contains(translate(normalize-space(string(.)),'TIMING','timing'),'timing')]"
    ).first

    if timing_anchor.count():
        tbl = page.locator("xpath=(//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::strong or self::b]"
                           f"[contains(translate(normalize-space(string(.)),'TIMING','timing'),'timing')])[1]/following::table[1]")
        if tbl.count():
            return table_to_records(tbl.first)

    # 3) Fall back: pick the table that contains the word Timing anywhere
    for i in range(n):
        t = tables.nth(i)
        if target in _normalize(t.inner_text()):
            return table_to_records(t)

    raise RuntimeError("Could not find a table named 'Timing' on the page.")


def table_to_records(table_locator):
    # Extract headers
    headers = table_locator.locator("thead tr th")
    if headers.count() == 0:
        # Some tables have headers in first row of tbody
        first_row_cells = table_locator.locator("tr").first.locator("th, td")
        hdrs = [c.inner_text().strip() for c in first_row_cells.all()]
        data_rows = table_locator.locator("tr").nth(1).locator("..")  # dummy
        # We'll just parse all rows and treat first as header
        rows = table_locator.locator("tr").all()
        matrix = []
        for r in rows:
            cells = r.locator("th, td").all()
            matrix.append([c.inner_text().strip() for c in cells])
        if not matrix:
            return []
        hdrs = matrix[0]
        body = matrix[1:]
    else:
        hdrs = [h.inner_text().strip() for h in headers.all()]
        # Extract body rows
        rows = table_locator.locator("tbody tr").all()
        body = []
        for r in rows:
            cells = r.locator("td, th").all()
            body.append([c.inner_text().strip() for c in cells])

    # Normalize row width
    out = []
    for row in body:
        row = row + [""] * (len(hdrs) - len(row))
        row = row[: len(hdrs)]
        out.append(dict(zip(hdrs, row)))
    return out


def login(page, email, password, timeout_ms=30000):
    """
    Login to www.vectorvest.com/vvlogin/login.aspx
    This is an ASP.NET form with specific field names.
    """
    page.wait_for_load_state("domcontentloaded")

    # Wait for password field to appear
    try:
        page.wait_for_selector("input[type='password']", timeout=10000)
        print("Login form loaded")
    except PWTimeout:
        print("No login form detected - may already be logged in")
        return

    # VectorVest login.aspx uses standard ASP.NET form fields
    # Try common ASP.NET username field names
    username_selectors = [
        "input[name*='txtUserName' i]",
        "input[name*='username' i]",
        "input[name*='email' i]",
        "input[id*='txtUserName' i]",
        "input[id*='username' i]",
        "input[type='text']",
        "input[type='email']",
    ]
    
    username_field = None
    for sel in username_selectors:
        loc = page.locator(sel).first
        if loc.count():
            username_field = loc
            break
    
    if username_field:
        username_field.fill(email)
        print(f"Filled username: {email}")
    else:
        print("WARNING: Could not find username field")

    # Fill password
    password_field = page.locator("input[type='password']").first
    password_field.fill(password)
    print("Filled password")

    # Click login button - ASP.NET forms often use input[type='submit'] or specific button IDs
    login_btn_selectors = [
        "input[type='submit']",
        "button[type='submit']",
        "input[name*='btnLogin' i]",
        "input[id*='btnLogin' i]",
        "button:has-text('Login')",
        "button:has-text('Sign in')",
        "input[value*='Login' i]",
    ]
    
    clicked = False
    for sel in login_btn_selectors:
        loc = page.locator(sel).first
        if loc.count():
            print(f"Clicking login button: {sel}")
            loc.click()
            clicked = True
            break
    
    if not clicked:
        print("No login button found, pressing Enter in password field...")
        password_field.press("Enter")

    # Wait for navigation / authenticated load
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
        print(f"Login completed, current URL: {page.url}")
        
        # Check for error messages
        error_selectors = [
            "#login_error",
            ".error",
            ".login-error",
            "[class*='error']",
            "[id*='error']",
        ]
        for sel in error_selectors:
            error_elem = page.locator(sel).first
            if error_elem.count() and error_elem.is_visible():
                error_text = error_elem.inner_text()
                if error_text.strip():
                    print(f"⚠️  Login error detected: {error_text.strip()}")
                    break
        
    except PWTimeout:
        print("Timeout waiting for page load after login (may be okay if slow)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://views-us.vectorvest.com/weekly_newsletters/12653-revision-v1/")
    ap.add_argument("--table", default="timing", help="Table name to extract (default: timing)")
    ap.add_argument("--out", default="timing_table", help="Output prefix (default: timing_table)")
    ap.add_argument("--headful", action="store_true", help="Run with visible browser for debugging")
    ap.add_argument("--debug-login", action="store_true", help="Debug login flow and exit")
    ap.add_argument("--cookies-path", help="Path to JSON cookies exported from browser (optional)")
    ap.add_argument("--feed-url", default="https://views-us.vectorvest.com/category/weekly-newsletter/feed/", help="RSS feed to resolve the latest newsletter")
    ap.add_argument("--latest-from-feed", action="store_true", help="Pull the most recent entry link from --feed-url and override --url")
    args = ap.parse_args()

    email = os.getenv("VECTORVEST_EMAIL")
    password = os.getenv("VECTORVEST_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "Set env vars VECTORVEST_EMAIL and VECTORVEST_PASSWORD before running."
        )

    out_csv = Path(f"{args.out}.csv")
    out_json = Path(f"{args.out}.json")

    target_url = args.url
    if args.latest_from_feed:
        target_url = get_latest_link_from_feed(args.feed_url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        ctx = browser.new_context()
        if args.cookies_path:
            load_cookies_into_context(ctx, args.cookies_path)
        page = ctx.new_page()

        # Navigate to VectorVest login page
        login_url = "https://www.vectorvest.com/vvlogin/login.aspx"
        print(f"Navigating to {login_url}...")
        page.goto(login_url, wait_until="domcontentloaded")

        # Login
        login(page, email, password)

        if args.debug_login:
            print("\nDebug mode: logged in, stopping here")
            ctx.close()
            browser.close()
            return

        # Navigate to newsletter URL
        print(f"\n=== Attempting to access newsletter ===")
        print(f"Target URL: {target_url}")
        page.goto(target_url, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        
        # Debug current state
        current_url = page.url
        print(f"Landed on: {current_url}")
        
        # Check for common indicators we're not on the right page
        page_title = page.title()
        print(f"Page title: {page_title}")
        
        # Look for password field (indicates login page)
        has_password_field = page.locator("input[type='password']").count() > 0
        print(f"Has password field: {has_password_field}")
        
        # Look for tables
        table_count = page.locator("table").count()
        print(f"Tables found: {table_count}")
        
        # Check if we landed on a redirect/login page
        if "redirect_to=" in current_url or "login" in current_url.lower():
            print(f"\n⚠️  Detected redirect/login page")
            print(f"This means we're not authenticated on views-us.vectorvest.com")
            print(f"The login to www.vectorvest.com doesn't carry over to views-us subdomain")
            
            if has_password_field:
                print("Attempting login on Views subdomain...")
                login(page, email, password)
                page.goto(target_url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                print(f"After retry, landed on: {page.url}")
            else:
                print("\n=== Searching for Views login endpoint ===")
                
                # Look for login/sign-in links
                login_links = page.locator("a:has-text('Log in'), a:has-text('Login'), a:has-text('Sign in'), a:has-text('Sign In'), a[href*='login']").all()
                print(f"Found {len(login_links)} potential login links:")
                for i, link in enumerate(login_links[:10]):  # Limit to first 10
                    try:
                        href = link.get_attribute("href")
                        text = link.inner_text().strip()
                        print(f"  [{i}] '{text}' -> {href}")
                    except:
                        pass
                
                # Check for any forms on the page
                forms = page.locator("form").all()
                print(f"\nFound {len(forms)} forms on page:")
                for i, form in enumerate(forms[:5]):
                    try:
                        action = form.get_attribute("action")
                        method = form.get_attribute("method")
                        print(f"  Form {i}: action={action}, method={method}")
                    except:
                        pass
                
                # Look for any auth-related URLs in the page source
                html_content = page.content()
                import re
                auth_urls = re.findall(r'https?://[^\s<>"]+(?:login|auth|signin)[^\s<>"]*', html_content, re.IGNORECASE)
                unique_auth_urls = list(set(auth_urls[:10]))  # Dedupe and limit
                if unique_auth_urls:
                    print(f"\nFound {len(unique_auth_urls)} auth-related URLs in page source:")
                    for url in unique_auth_urls:
                        print(f"  {url}")
                
                # Try standard WordPress login endpoints
                print("\n=== Trying standard WordPress login endpoints ===")
                wp_login_urls = [
                    "https://views-us.vectorvest.com/wp-login.php",
                    "https://views-us.vectorvest.com/wp-admin/",
                ]
                
                for login_url in wp_login_urls:
                    print(f"\nTrying: {login_url}")
                    page.goto(login_url, wait_until="domcontentloaded")
                    page.wait_for_timeout(1000)
                    
                    if page.locator("input[type='password']").count() > 0:
                        print(f"✓ Found login form at {login_url}")
                        print("Attempting login...")
                        login(page, email, password)
                        
                        # Check cookies after login
                        cookies = page.context.cookies()
                        views_cookies = [c for c in cookies if 'views-us.vectorvest.com' in c.get('domain', '')]
                        print(f"\nCookies for views-us.vectorvest.com after login: {len(views_cookies)}")
                        for cookie in views_cookies[:5]:  # Show first 5
                            print(f"  {cookie['name']}: {cookie['value'][:50] if len(cookie['value']) > 50 else cookie['value']}")
                        
                        # After login, try the newsletter URL again
                        print(f"Navigating to newsletter after WP login...")
                        page.goto(target_url, wait_until="domcontentloaded")
                        page.wait_for_timeout(2000)
                        final_url = page.url
                        print(f"Final URL: {final_url}")
                        
                        # Check if we're still on a redirect page
                        if "redirect_to=" in final_url:
                            print("Still on redirect page - attempting to follow redirect_to parameter...")
                            # Extract and follow the redirect_to parameter
                            import urllib.parse
                            parsed = urllib.parse.urlparse(final_url)
                            params = urllib.parse.parse_qs(parsed.query)
                            if 'redirect_to' in params:
                                redirect_path = params['redirect_to'][0]
                                base_url = f"{parsed.scheme}://{parsed.netloc}"
                                target_url = base_url + redirect_path
                                print(f"Following redirect to: {target_url}")
                                page.goto(target_url, wait_until="domcontentloaded")
                                page.wait_for_timeout(2000)
                                print(f"After following redirect: {page.url}")
                        
                        break
                    else:
                        print(f"✗ No login form at {login_url}")
                
                print("\nIf WordPress login didn't work, recommendation: export cookies from browser after manual login")

        # Give the newsletter content a moment to render
        page.wait_for_timeout(2000)

        # Debug: save screenshot and HTML
        page.screenshot(path="debug_page.png")
        Path("debug_page.html").write_text(page.content(), encoding="utf-8")
        print(f"\n=== Debug output ===")
        print(f"Saved: debug_page.png")
        print(f"Saved: debug_page.html")
        print(f"Final URL: {page.url}")
        print(f"Tables on page: {page.locator('table').count()}")

        records = extract_table_by_name(page, target_name=args.table)
        df = pd.DataFrame(records)

        df.to_csv(out_csv, index=False)
        out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

        print(f"Wrote {out_csv} ({len(df)} rows, {len(df.columns)} cols)")
        print(f"Wrote {out_json}")

        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
