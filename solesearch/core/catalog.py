"""Sneaker facts from catalog.json and generated buy links."""
import json
from pathlib import Path
from urllib.parse import quote_plus

# Search pages, so links never go stale when a product page is removed.
SEARCH_URLS = {
    "Nike": "https://www.nike.com/w?q={q}",
    "Jordan": "https://www.nike.com/w?q={q}",
    "Adidas": "https://www.adidas.com/us/search?q={q}",
    "New Balance": "https://www.newbalance.com/search?q={q}",
    "ASICS": "https://www.asics.com/us/en-us/search?q={q}",
}
RESALE_URLS = {
    "StockX": "https://stockx.com/search?s={q}",
    "GOAT": "https://www.goat.com/search?query={q}",
}


class Catalog:
    def __init__(self, path: Path):
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def model(self, model_key):
        """Basic info for a model; falls back to a readable name from the folder name."""
        info = self.data.get(model_key, {})
        return {
            "brand": info.get("brand", ""),
            "name": info.get("name", model_key.replace("_", " ").title()),
            "description": info.get("description", ""),
        }

    def colorway(self, model_key, colorway_key):
        """Facts for one colorway. Missing facts stay None and the app shows 'not verified'."""
        cw = self.data.get(model_key, {}).get("colorways", {}).get(colorway_key, {})
        return {
            "name": cw.get("name") or colorway_key.replace("_", " ").title(),
            "sku": cw.get("sku"),
            "release_date": cw.get("release_date"),
            "retail_price_usd": cw.get("retail_price_usd"),
            "source": cw.get("source"),
        }

    def all_models(self):
        return [k for k in self.data if not k.startswith("_")]  # "_" keys are notes

    def colorway_keys(self, model_key):
        return list(self.data.get(model_key, {}).get("colorways", {}).keys())

    def buy_links(self, model_key, colorway_key=None):
        m = self.model(model_key)
        query = m["name"]
        if colorway_key:
            query += " " + self.colorway(model_key, colorway_key)["name"]
        q = quote_plus(query)
        links = {}
        if m["brand"] in SEARCH_URLS:
            label = "Nike" if m["brand"] == "Jordan" else m["brand"]
            links[label] = SEARCH_URLS[m["brand"]].format(q=q)
        for site, url in RESALE_URLS.items():
            links[site] = url.format(q=q)
        return links
