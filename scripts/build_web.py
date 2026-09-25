"""Build the Vercel frontend: web/index.html (the hub page) and web/vercel.json forwarding /api and /research
to the Railway backend. Usage: `uv run python scripts/build_web.py https://<your-app>.up.railway.app`.

The page is the same file the backend serves (packages/api/.../dashboard.html); Vercel forwards its API
calls server-side, so the browser sees one origin and no CORS setup is needed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "packages" / "api" / "src" / "autotrader" / "api" / "dashboard.html"
WEB = ROOT / "web"


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].startswith("https://"):
        print("usage: build_web.py https://<backend host>")
        return 2
    api = sys.argv[1].rstrip("/")
    WEB.mkdir(exist_ok=True)
    shutil.copyfile(PAGE, WEB / "index.html")
    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "rewrites": [
            {"source": "/api/:path*", "destination": f"{api}/api/:path*"},
            {"source": "/research/:path*", "destination": f"{api}/research/:path*"},
        ],
        "headers": [{"source": "/(.*)", "headers": [{"key": "X-Frame-Options", "value": "DENY"}]}],
    }
    (WEB / "vercel.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"web/ ready: the page forwards /api and /research to {api}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
