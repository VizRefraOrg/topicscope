"""Rebuild static/blog/index.html from static/blog/*.html for blog.vizrefra.com.

Scans each post for <title>, <meta name="description">, and
<meta property="article:published_time"> and emits a chronological index page.

Run during the CI deploy (see .github/workflows/main_topicscope-api.yml) so the
prebuilt index ships in the deploy artifact and the FastAPI root handler can
just FileResponse it on blog.vizrefra.com requests.
"""
import re
from datetime import datetime
from pathlib import Path

BLOG_DIR = Path(__file__).resolve().parent.parent / "static" / "blog"
INDEX_PATH = BLOG_DIR / "index.html"
SITE_URL = "https://blog.vizrefra.com"

_TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)
_DESC_RE = re.compile(r'<meta\s+name="description"\s+content="([^"]+)"', re.IGNORECASE)
_PUB_RE = re.compile(r'<meta\s+property="article:published_time"\s+content="([^"]+)"', re.IGNORECASE)


def scan_posts():
    if not BLOG_DIR.exists():
        return []
    posts = []
    for p in BLOG_DIR.glob("*.html"):
        if p.name == "index.html":
            continue
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        title_m = _TITLE_RE.search(html)
        title = title_m.group(1).strip() if title_m else p.stem
        desc_m = _DESC_RE.search(html)
        desc = desc_m.group(1).strip() if desc_m else ""
        pub_m = _PUB_RE.search(html)
        pub = pub_m.group(1).strip() if pub_m else ""
        posts.append({"slug": p.stem, "title": title, "desc": desc, "pub": pub})
    posts.sort(key=lambda x: x["pub"], reverse=True)
    return posts


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blog | VizRefra</title>
<meta name="description" content="VizRefra blog — insights for content teams and researchers visualising large text datasets.">
<link rel="canonical" href="{site_url}/">
<style>
body {{font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 760px; margin: 0 auto; padding: 4rem 1.5rem; color: #1a1a1a; line-height: 1.6;}}
h1 {{font-size: 2.5rem; margin-bottom: 2rem;}}
a {{color: #2563eb; text-decoration: none;}}
a:hover {{text-decoration: underline;}}
ul {{list-style: none; padding: 0;}}
.post-item {{padding: 1.5rem 0; border-bottom: 1px solid #e5e5e5;}}
.post-item h2 {{font-size: 1.5rem; margin: 0 0 .5rem;}}
.post-item p {{margin: 0 0 .5rem; color: #555;}}
.post-item time {{font-size: .85rem; color: #999;}}
header {{margin-bottom: 3rem;}}
header nav a {{margin-right: 1rem;}}
</style>
</head>
<body>
<header>
<nav>
<a href="https://vizrefra.com">VizRefra home</a>
<a href="{site_url}/">Blog</a>
</nav>
</header>
<main>
<h1>Blog</h1>
<ul>
{posts}
</ul>
</main>
</body>
</html>
"""

POST_ITEM_TEMPLATE = """<li class="post-item">
<a href="{site_url}/{slug}">
<h2>{title}</h2>
<p>{desc}</p>
<time datetime="{pub}">{pub_pretty}</time>
</a>
</li>"""


def render():
    posts = scan_posts()
    items = []
    for p in posts:
        pub_pretty = ""
        if p["pub"]:
            try:
                pub_pretty = datetime.fromisoformat(p["pub"].replace("Z", "+00:00")).strftime("%d %B %Y")
            except ValueError:
                pub_pretty = p["pub"]
        items.append(POST_ITEM_TEMPLATE.format(
            site_url=SITE_URL,
            slug=p["slug"],
            title=p["title"],
            desc=p["desc"],
            pub=p["pub"],
            pub_pretty=pub_pretty,
        ))
    body = "\n".join(items) or '<li class="post-item"><p>No posts yet.</p></li>'
    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(INDEX_TEMPLATE.format(site_url=SITE_URL, posts=body), encoding="utf-8")
    print(f"Wrote {INDEX_PATH} with {len(posts)} posts")


if __name__ == "__main__":
    render()
