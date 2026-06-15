"""Regenerate static/sitemap.xml with vizrefra.com + blog.vizrefra.com URLs.

Run during CI so the sitemap stays in sync with whatever blog posts ship in
static/blog/*.html.
"""
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = ROOT / "static" / "blog"
SITEMAP = ROOT / "static" / "sitemap.xml"
MAIN_URL = "https://vizrefra.com"
BLOG_URL = "https://blog.vizrefra.com"


def main():
    now = datetime.utcnow().strftime("%Y-%m-%d")
    urls = [
        (f"{MAIN_URL}/", 1.0, "weekly"),
        (f"{MAIN_URL}/about-jay", 0.6, "monthly"),
        (f"{BLOG_URL}/", 0.9, "weekly"),
    ]
    if BLOG_DIR.exists():
        for p in sorted(BLOG_DIR.glob("*.html")):
            if p.name == "index.html":
                continue
            urls.append((f"{BLOG_URL}/{p.stem}", 0.7, "monthly"))
    items = "\n".join(
        f"  <url><loc>{u}</loc><lastmod>{now}</lastmod><changefreq>{f}</changefreq><priority>{pr}</priority></url>"
        for u, pr, f in urls
    )
    SITEMAP.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{items}\n</urlset>\n',
        encoding="utf-8",
    )
    print(f"Wrote {SITEMAP} with {len(urls)} urls")


if __name__ == "__main__":
    main()
