"""Open the app on a phone: the public https address of this server, and a QR
code for any page of it.

The app runs on this Mac; phones reach it through a Cloudflare tunnel
(scripts/start_live.sh), because a phone camera only works over https. The
public address is, in order:
  1. PUBLIC_URL in the environment (a fixed domain, e.g. a named tunnel)
  2. the address the request itself came in on, when that is the tunnel
  3. what the running quick tunnel reports (its URL changes on every start)
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.request

import qrcode
import qrcode.image.svg

TUNNEL_METRICS = [f"http://127.0.0.1:{p}/quicktunnel" for p in range(20241, 20246)]
_SAFE_PATH = re.compile(r"^/(?!/)[^\s\\]{0,500}$")
_LOCAL = re.compile(r"^(localhost|127\.|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1\]?$)")
_cache = {"at": 0.0, "url": None}


def _quick_tunnel() -> str | None:
    """The running quick tunnel's address, looked up at most every 20 s."""
    if time.time() - _cache["at"] < 20:
        return _cache["url"]
    url = None
    for probe in TUNNEL_METRICS:
        try:
            with urllib.request.urlopen(probe, timeout=0.5) as r:
                host = json.loads(r.read()).get("hostname")
            if host:
                url = f"https://{host}"
                break
        except (OSError, ValueError):
            continue
    _cache.update(at=time.time(), url=url)
    return url


def public_base(host: str | None, forwarded_proto: str | None) -> str | None:
    """https base URL a phone can open, or None when the app isn't live."""
    fixed = os.environ.get("PUBLIC_URL", "").strip().rstrip("/")
    if fixed.startswith("https://"):
        return fixed
    host = (host or "").strip()
    if host and forwarded_proto == "https" and not _LOCAL.match(host):
        return f"https://{host}"
    return _quick_tunnel()


def safe_path(path: str) -> bool:
    """A path on this site ("/..."), never another site ("//evil.com")."""
    return bool(_SAFE_PATH.match(path or ""))


def qr_svg(text: str) -> str:
    buf = io.BytesIO()
    qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2).save(buf)
    return buf.getvalue().decode()
