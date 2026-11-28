"""Colorway matcher: compares a photo against reference photos in gallery/<model>/<colorway>/.

Score = 50% deep-feature similarity (shape/details) + 50% color-histogram similarity
(colors). The fine-tuned classifier learns that a "Bred" and a "Chicago" are both
Jordan 1s, so on its own it partly ignores color. The histogram puts color back in.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
EMB_WEIGHT = 0.5  # how much shape matters vs. color; tune this if matches look off


def color_histogram(img: Image.Image, bins=(12, 4, 4)) -> np.ndarray:
    """HSV color histogram of the central 80% of the image (trims some background)."""
    img = img.convert("RGB")
    w, h = img.size
    img = img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))).resize((128, 128))
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=bins, range=((0, 256), (0, 256), (0, 256)))
    hist = hist.flatten()
    return hist / (hist.sum() + 1e-8)


def _gallery_files(gallery_dir: Path):
    return sorted(p for p in gallery_dir.glob("*/*/*") if p.suffix.lower() in IMG_EXTS)


class ColorwayMatcher:
    def __init__(self, gallery_dir: Path, cache_path: Path, analyze_fn):
        """analyze_fn(img) -> (probs, embedding), i.e. SneakerModel.analyze."""
        self.gallery_dir = gallery_dir
        files = _gallery_files(gallery_dir)
        signature = [f"{p.relative_to(gallery_dir)}:{p.stat().st_mtime_ns}" for p in files]

        cached = self._load_cache(cache_path, signature)
        if cached is not None:
            self.models, self.colorways, self.paths, self.embs, self.hists = cached
            return

        models, colorways, paths, embs, hists = [], [], [], [], []
        for p in files:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue  # skip broken images
            _, emb = analyze_fn(img)
            models.append(p.parent.parent.name)
            colorways.append(p.parent.name)
            paths.append(str(p))
            embs.append(emb)
            hists.append(color_histogram(img))

        self.models, self.colorways, self.paths = models, colorways, paths
        self.embs = np.array(embs) if embs else np.zeros((0, 1280))
        self.hists = np.array(hists) if hists else np.zeros((0, 192))
        self._save_cache(cache_path, signature)

    # --- cache so the gallery is only processed again when its files change ---
    def _load_cache(self, cache_path, signature):
        meta_path = cache_path.with_suffix(".json")
        if not (cache_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") != signature:
            return None
        arrays = np.load(cache_path)
        return meta["models"], meta["colorways"], meta["paths"], arrays["embs"], arrays["hists"]

    def _save_cache(self, cache_path, signature):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embs=self.embs, hists=self.hists)
        cache_path.with_suffix(".json").write_text(json.dumps({
            "signature": signature, "models": self.models,
            "colorways": self.colorways, "paths": self.paths,
        }))

    def __len__(self):
        return len(self.paths)

    def match(self, img: Image.Image, emb: np.ndarray, model_key=None, k=3):
        """Best-matching colorways. model_key=None searches every model ("similar sneakers")."""
        if not self.paths:
            return []
        hist = color_histogram(img)
        emb_sim = self.embs @ emb                              # cosine similarity, -1..1
        emb_sim = (emb_sim + 1) / 2                            # rescale to 0..1
        hist_sim = np.minimum(self.hists, hist).sum(axis=1)    # histogram intersection, 0..1
        scores = EMB_WEIGHT * emb_sim + (1 - EMB_WEIGHT) * hist_sim

        best = {}  # (model, colorway) -> (score, reference photo)
        for i, s in enumerate(scores):
            if model_key and self.models[i] != model_key:
                continue
            key = (self.models[i], self.colorways[i])
            if key not in best or s > best[key][0]:
                best[key] = (float(s), self.paths[i])

        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
        return [{"model": m, "colorway": c, "score": s, "photo": p} for (m, c), (s, p) in ranked]

    def colorways_for(self, model_key):
        return sorted({c for m, c in zip(self.models, self.colorways) if m == model_key})
