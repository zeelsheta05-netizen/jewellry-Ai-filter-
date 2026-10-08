"""The "On phone" link: which public address a phone gets, and which paths are allowed."""
import pytest

from jewelsearch import phone


@pytest.fixture(autouse=True)
def no_tunnel(monkeypatch):
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    monkeypatch.setattr(phone, "_quick_tunnel", lambda: None)


def test_fixed_public_url_wins(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://finder.example.com/")
    assert phone.public_base("abc.trycloudflare.com", "https") == "https://finder.example.com"


def test_request_through_the_tunnel_uses_its_own_address():
    assert phone.public_base("abc.trycloudflare.com", "https") == "https://abc.trycloudflare.com"


@pytest.mark.parametrize("host", ["localhost:8765", "127.0.0.1:8765", "0.0.0.0:8765", "192.168.1.5:8765", "10.0.0.6:8765"])
def test_local_addresses_are_never_given_to_a_phone(host, monkeypatch):
    assert phone.public_base(host, "https") is None
    monkeypatch.setattr(phone, "_quick_tunnel", lambda: "https://live.trycloudflare.com")
    assert phone.public_base(host, None) == "https://live.trycloudflare.com"


def test_plain_http_is_not_public():
    assert phone.public_base("abc.trycloudflare.com", "http") is None


@pytest.mark.parametrize("path,ok", [("/", True), ("/?q=gold%20ring", True), ("/tryon?model=1347-c6282a", True),
                                     ("//evil.com", False), ("https://evil.com", False), ("/a b", False), ("", False)])
def test_only_paths_on_this_site(path, ok):
    assert phone.safe_path(path) is ok


def test_qr_code_is_svg():
    assert phone.qr_svg("https://abc.trycloudflare.com/").lstrip().startswith("<?xml")
