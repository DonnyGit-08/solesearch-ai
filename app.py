"""SoleSearch AI: identify sneakers from a photo and keep a digital closet.

Run with:  streamlit run app.py
"""
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from catalog import Catalog
from closet import Closet

ROOT = Path(__file__).parent
# Model files can live in a "models" folder or at the top level of the repo
MODELS_DIR = ROOT / "models" if (ROOT / "models" / "solesearch_effnet_b0.pt").exists() else ROOT
MODEL_PATH = MODELS_DIR / "solesearch_effnet_b0.pt"
GALLERY_DIR = ROOT / "gallery"
DATA_DIR = ROOT / "data"
CONDITIONS = ["New / deadstock", "Like new", "Used - good", "Used - well worn"]
UNKNOWN = "__unknown__"
OTHER = "__other__"

st.set_page_config(page_title="SoleSearch AI", page_icon="👟", layout="wide")

# ---------- look and feel: dark, photo-first, SNKRS-inspired ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0A0A0B; --surface: #141416; --surface2: #1C1C21; --line: rgba(255,255,255,.08);
  --text: #F5F5F7; --muted: #8E8A99; --accent: #8B5CF6; --accent2: #A78BFA; --glow: rgba(139,92,246,.45);
}
.stApp { background: var(--bg); }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Inter Tight', 'Inter', sans-serif !important; font-weight: 800 !important;
  letter-spacing: -0.02em; text-transform: uppercase; }
footer, header [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 1.6rem !important; padding-bottom: 140px !important; max-width: 1180px; }

@keyframes ssUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes ssGrow { from { width: 0; } }
@keyframes ssSweep { 0% { top: -40%; } 100% { top: 105%; } }
@keyframes ssPulse { 0%,100% { opacity: .55; } 50% { opacity: 1; } }
@keyframes ssFadeOut { to { opacity: 0; } }

/* brand bar */
.ss-topbar { display: flex; align-items: center; justify-content: space-between; margin: 0 0 18px; }
.ss-logo { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 900; font-size: 1.35rem; letter-spacing: -.03em; color: var(--text); }
.ss-logo span { color: var(--accent2); }
.ss-pill { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
  border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; }
.ss-pill b { color: var(--text); }

/* hero */
.ss-hero { position: relative; overflow: hidden; border-radius: 28px; padding: 54px 44px 44px; margin-bottom: 30px;
  background:
    radial-gradient(600px 300px at 88% 10%, rgba(139,92,246,.55), transparent 60%),
    radial-gradient(500px 260px at 0% 120%, rgba(91,33,182,.35), transparent 60%),
    linear-gradient(180deg, #121016, #0B0A0E);
  border: 1px solid var(--line); animation: ssUp .6s ease; }
.ss-hero::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 44px 44px; mask-image: linear-gradient(90deg, transparent, #000 60%); -webkit-mask-image: linear-gradient(90deg, transparent, #000 60%); }
.ss-kicker { font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--accent2); }
.ss-hero h1 { color: #FFFFFF !important; font-size: clamp(2.3rem, 6vw, 4.4rem) !important; line-height: .95 !important;
  margin: 14px 0 16px !important; padding: 0 !important; letter-spacing: -.035em !important; }
.ss-hero h1 .accent { color: var(--accent2); }
.ss-hero p { color: #B7B2C4; font-size: 1.05rem; max-width: 520px; margin: 0; line-height: 1.55; }
.ss-hero.compact { padding: 38px 36px 32px; }
.ss-hero.compact h1 { font-size: clamp(2rem, 5vw, 3.2rem) !important; }
.ss-steps { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 26px; position: relative; z-index: 1; }
.ss-step { border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04); border-radius: 999px;
  padding: 8px 14px; font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #E7E3F0; }
.ss-step b { color: var(--accent2); margin-right: 8px; }

/* section titles */
.ss-section { margin: 38px 0 14px; }
.ss-section .t { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 800; font-size: 1.7rem; text-transform: uppercase;
  letter-spacing: -.02em; color: var(--text); line-height: 1.05; margin-top: 6px; }

/* feed cards (photo-first) */
.ss-feed { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }
.ss-tile { position: relative; border-radius: 22px; overflow: hidden; background: var(--surface);
  border: 1px solid var(--line); aspect-ratio: 4 / 5; animation: ssUp .55s ease both;
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.ss-tile:hover { transform: translateY(-4px); border-color: rgba(167,139,250,.5); box-shadow: 0 18px 40px rgba(0,0,0,.5), 0 0 0 1px rgba(167,139,250,.25); }
.ss-tile img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transition: transform .5s ease; }
.ss-tile:hover img { transform: scale(1.05); }
.ss-tile .ph { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 900; font-size: 3.2rem; color: rgba(255,255,255,.07); letter-spacing: -.04em;
  background: radial-gradient(circle at 50% 40%, #24202E, #121115); }
.ss-tile .cap { position: absolute; left: 0; right: 0; bottom: 0; padding: 48px 18px 16px;
  background: linear-gradient(180deg, transparent, rgba(0,0,0,.85)); }
.ss-tile .brand { font-size: .68rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--accent2); }
.ss-tile .name { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 800; font-size: 1.2rem; text-transform: uppercase; color: #FFF; margin-top: 4px; line-height: 1.05; }

/* scanner frame around the user's photo */
.ss-scan { position: relative; border-radius: 26px; overflow: hidden; background: var(--surface); border: 1px solid var(--line); animation: ssUp .5s ease; }
.ss-scan img { display: block; width: 100%; height: auto; }
.ss-scan .sweep { position: absolute; left: 0; right: 0; height: 40%; top: -40%;
  background: linear-gradient(180deg, transparent, rgba(139,92,246,.30) 80%, rgba(196,181,253,.9) 100%);
  animation: ssSweep 1.4s cubic-bezier(.4,0,.2,1) 2, ssFadeOut .3s ease 2.8s forwards; }
.ss-scan .c { position: absolute; width: 28px; height: 28px; border: 2px solid var(--accent2); }
.ss-scan .tl { top: 14px; left: 14px; border-right: 0; border-bottom: 0; border-radius: 10px 0 0 0; }
.ss-scan .tr { top: 14px; right: 14px; border-left: 0; border-bottom: 0; border-radius: 0 10px 0 0; }
.ss-scan .bl { bottom: 14px; left: 14px; border-right: 0; border-top: 0; border-radius: 0 0 0 10px; }
.ss-scan .br { bottom: 14px; right: 14px; border-left: 0; border-top: 0; border-radius: 0 0 10px 0; }
.ss-scan .tag { position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%); white-space: nowrap;
  background: rgba(10,10,11,.75); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,.12); border-radius: 999px; padding: 7px 14px;
  font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #E7E3F0; }
.ss-scan .tag i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #4ADE80;
  margin-right: 8px; box-shadow: 0 0 10px #4ADE80; animation: ssPulse 1.6s ease infinite; }

/* result + detail cards */
.ss-card { background: var(--surface); color: var(--text); border: 1px solid var(--line); border-radius: 26px;
  padding: 28px; animation: ssUp .55s ease; margin-bottom: 14px; }
.ss-card.glow { background: radial-gradient(400px 200px at 100% 0%, rgba(139,92,246,.18), transparent 70%), var(--surface); }
.ss-badge { display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px; border-radius: 999px;
  font-weight: 700; font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; }
.ss-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }
.ss-name { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 900; font-size: clamp(1.9rem, 4vw, 2.8rem);
  text-transform: uppercase; letter-spacing: -.03em; line-height: .98; margin: 16px 0 8px; color: #FFF; }
.ss-sub { color: var(--muted); font-size: .95rem; margin: 0; }
.ss-desc { color: #C9C5D3; margin: 14px 0 0; line-height: 1.6; }
.ss-label { font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .16em; color: var(--muted); }

.ss-bar-row { margin: 14px 0 0; }
.ss-bar-label { display: flex; justify-content: space-between; font-weight: 600; font-size: .9rem; color: var(--text); }
.ss-bar-label span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
.ss-bar { height: 6px; background: rgba(255,255,255,.07); border-radius: 999px; overflow: hidden; margin-top: 8px; }
.ss-bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6D28D9, #A78BFA);
  box-shadow: 0 0 14px var(--glow); animation: ssGrow 1.1s cubic-bezier(.2,.8,.2,1) .2s both; }
.ss-bar.dim > div { background: rgba(255,255,255,.22); box-shadow: none; }

.ss-detail { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 0; overflow: hidden; padding: 0; }
.ss-detail .img { position: relative; min-height: 260px; background: #111; }
.ss-detail .img img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.ss-detail .body { padding: 30px; }
@media (max-width: 760px) { .ss-detail { grid-template-columns: 1fr; } .ss-hero { padding: 40px 26px 32px; } }
@media (max-width: 640px) { .ss-feed { grid-template-columns: 1fr 1fr; gap: 10px; }
  .ss-tile { border-radius: 18px; } .ss-tile .name { font-size: .95rem; } .ss-tile .cap { padding: 36px 12px 12px; }
  .ss-tile .ph { font-size: 2.2rem; } }

.ss-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 12px 0 4px; }
.ss-stat { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px; animation: ssUp .5s ease; }
.ss-stat .k { font-size: .66rem; text-transform: uppercase; letter-spacing: .16em; color: var(--muted); font-weight: 700; }
.ss-stat .v { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 800; font-size: 1.5rem; margin-top: 6px; color: #FFF; letter-spacing: -.02em; }
.ss-stat .v.muted { font-family: 'Inter', sans-serif; font-size: .95rem; color: #5F5B6B; font-weight: 600; letter-spacing: 0; }

.ss-tips { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.ss-tip { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; padding: 20px; animation: ssUp .5s ease both; }
.ss-tip .n { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 900; font-size: 1.6rem; color: var(--accent2); letter-spacing: -.03em; }
.ss-tip .t { font-weight: 700; margin-top: 10px; color: #FFF; text-transform: uppercase; font-size: .86rem; letter-spacing: .06em; }
.ss-tip .d { color: var(--muted); font-size: .9rem; margin-top: 6px; line-height: 1.5; }

.ss-empty { text-align: center; padding: 64px 24px; border-radius: 26px; border: 1px dashed rgba(167,139,250,.35);
  background: radial-gradient(400px 200px at 50% 0%, rgba(139,92,246,.14), transparent 70%), var(--surface); animation: ssUp .5s ease; }
.ss-empty .t { font-family: 'Inter Tight', 'Inter', sans-serif; font-weight: 900; font-size: 1.8rem; text-transform: uppercase; color: #FFF; letter-spacing: -.02em; margin-top: 10px; }
.ss-empty .d { color: var(--muted); margin-top: 8px; }

/* Streamlit widgets, restyled */
[data-testid="stImage"] img { border-radius: 18px; }
[data-testid="stFileUploaderDropzone"] { background: radial-gradient(300px 120px at 50% 0%, rgba(139,92,246,.16), transparent 70%), var(--surface) !important;
  border: 1.5px dashed rgba(167,139,250,.55) !important; border-radius: 24px !important; padding: 34px !important; }
[data-testid="stFileUploaderDropzone"] button { border-radius: 999px !important; }
div.stButton > button, [data-testid="stLinkButton"] a { border-radius: 999px !important; font-weight: 700 !important;
  text-transform: uppercase; letter-spacing: .06em; font-size: .78rem !important; border: 1px solid rgba(255,255,255,.18) !important;
  background: transparent !important; color: #FFF !important; transition: all .18s ease; padding: .55rem 1.1rem !important; }
div.stButton > button:hover, [data-testid="stLinkButton"] a:hover { background: #FFF !important; color: #0A0A0B !important; border-color: #FFF !important; }
[data-testid="stFormSubmitButton"] button { background: linear-gradient(135deg, #8B5CF6, #6D28D9) !important; color: #FFF !important;
  border: none !important; border-radius: 999px !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: .08em;
  padding: .75rem 1.8rem !important; box-shadow: 0 10px 30px rgba(139,92,246,.4) !important; }
[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line) !important; border-radius: 26px !important; padding: 26px !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 22px !important; border-color: var(--line) !important; background: var(--surface); }
[data-testid="stWidgetLabel"] p { font-size: .72rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: .12em; color: var(--muted) !important; }
[data-testid="stExpander"] details { border-radius: 20px !important; border-color: var(--line) !important; background: var(--surface); }

/* bottom tab bar */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.st-key-bottomnav { position: fixed; left: 50%; transform: translateX(-50%); bottom: 18px; z-index: 1000;
  width: min(440px, calc(100% - 28px)); padding: 7px; border-radius: 26px;
  background: rgba(20,20,23,.82); backdrop-filter: blur(20px) saturate(140%); -webkit-backdrop-filter: blur(20px) saturate(140%);
  border: 1px solid rgba(255,255,255,.09); box-shadow: 0 20px 50px rgba(0,0,0,.6); }
.st-key-bottomnav [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 4px !important; }
.st-key-bottomnav [data-testid="stColumn"], .st-key-bottomnav [data-testid="column"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
.st-key-bottomnav button { width: 100% !important; border: none !important; border-radius: 20px !important;
  background: transparent !important; color: #8E8A99 !important; padding: 11px 4px !important; box-shadow: none !important; }
.st-key-bottomnav button:hover { color: #FFF !important; background: rgba(255,255,255,.06) !important; }
.st-key-bottomnav button[kind="primary"], .st-key-bottomnav button[data-testid="stBaseButton-primary"] {
  background: #FFF !important; color: #0A0A0B !important; }
.st-key-bottomnav button p { font-size: .74rem !important; font-weight: 800 !important; letter-spacing: .1em; }
</style>
""", unsafe_allow_html=True)


# ---------- small UI helpers ----------
def esc(text):
    return html.escape(str(text))


def section(kicker, title):
    st.markdown(f'<div class="ss-section"><div class="ss-kicker">{esc(kicker)}</div>'
                f'<div class="t">{esc(title)}</div></div>', unsafe_allow_html=True)


def hero(kicker, title_html, subtitle, steps=None, compact=False):
    steps_html = ""
    if steps:
        steps_html = '<div class="ss-steps">' + "".join(
            f'<div class="ss-step"><b>0{i}</b>{esc(s)}</div>' for i, s in enumerate(steps, 1)) + "</div>"
    st.markdown(f'<div class="ss-hero{" compact" if compact else ""}"><div class="ss-kicker">{esc(kicker)}</div>'
                f'<h1>{title_html}</h1><p>{esc(subtitle)}</p>{steps_html}</div>', unsafe_allow_html=True)


def stat_tiles(items):
    """items: list of (label, value). None values show as 'Not verified'."""
    tiles = []
    for label, value in items:
        v = ('<div class="v muted">Not verified</div>' if value is None
             else f'<div class="v">{esc(value)}</div>')
        tiles.append(f'<div class="ss-stat"><div class="k">{esc(label)}</div>{v}</div>')
    st.markdown(f'<div class="ss-stats">{"".join(tiles)}</div>', unsafe_allow_html=True)


def confidence_badge(p, threshold):
    if p >= 0.85:
        return "Locked in", "#4ADE80", "rgba(74,222,128,.12)"
    if p >= threshold:
        return "Likely match", "#FBBF24", "rgba(251,191,36,.12)"
    return "Not sure", "#F87171", "rgba(248,113,113,.12)"


def pil_b64(img, max_side=1000):
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def file_b64(path_str, mtime):  # mtime in the key: a replaced photo shows up right away
    return pil_b64(Image.open(path_str).convert("RGB"), max_side=800)


# ---------- shared resources (loaded once, then reused across reruns) ----------
@st.cache_resource(show_spinner="Warming up the AI...")
def load_model():
    from model import SneakerModel  # imported here so the closet works even without torch
    return SneakerModel(MODEL_PATH)


@st.cache_resource(show_spinner="Indexing colorway reference photos...")
def load_matcher(_model):
    from matcher import ColorwayMatcher
    return ColorwayMatcher(GALLERY_DIR, DATA_DIR / "gallery_index.npz", _model.analyze)


@st.cache_resource
def load_closet():
    return Closet(DATA_DIR / "closet.db", DATA_DIR / "photos")


catalog = Catalog(ROOT / "catalog.json")  # re-read every run, so catalog edits show up immediately
closet = load_closet()


def model_name(key):
    return catalog.model(key)["name"]


def fmt_price(p):
    return f"${p:,.0f}" if p is not None else None


def display_image(key):
    """A real photo for each sneaker model: displays/<key>.jpg or <key>.jpg at the top of the repo."""
    for folder in (ROOT / "displays", ROOT):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = folder / f"{key}{ext}"
            if path.exists():
                return path
    return None


def display_img_tag(key):
    path = display_image(key)
    if not path:
        return None
    return f'<img src="data:image/jpeg;base64,{file_b64(str(path), path.stat().st_mtime)}" alt="">'


def sneaker_feed(class_names):
    tiles = []
    for i, key in enumerate(class_names):
        m = catalog.model(key)
        img = display_img_tag(key)
        initials = "".join(w[0] for w in m["name"].split()[:3]).upper()
        media = img or f'<div class="ph">{esc(initials)}</div>'
        tiles.append(f'<div class="ss-tile" style="animation-delay:{i * 60}ms">{media}'
                     f'<div class="cap"><div class="brand">{esc(m["brand"])}</div>'
                     f'<div class="name">{esc(m["name"])}</div></div></div>')
    st.markdown(f'<div class="ss-feed">{"".join(tiles)}</div>', unsafe_allow_html=True)


# ---------- page: scan ----------
def details_section(model_key, colorway_key):
    m = catalog.model(model_key)
    cw = catalog.colorway(model_key, colorway_key) if colorway_key else None
    title = m["name"] + (f' “{cw["name"]}”' if cw else "")
    desc = f'<p class="ss-desc">{esc(m["description"])}</p>' if m["description"] else ""
    img = display_img_tag(model_key)
    body = (f'<div class="ss-kicker">{esc(m["brand"] or "Sneaker")}</div>'
            f'<div class="ss-name">{esc(title)}</div>{desc}')
    if img:
        st.markdown(f'<div class="ss-card ss-detail"><div class="img">{img}</div>'
                    f'<div class="body">{body}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ss-card glow">{body}</div>', unsafe_allow_html=True)

    items = []
    if cw:
        items += [("Release date", cw["release_date"]), ("Retail price", fmt_price(cw["retail_price_usd"]))]
        if cw["sku"]:
            items.append(("Style code", cw["sku"]))
    items.append(("Brand", m["brand"] or "—"))
    stat_tiles(items)
    if not cw:
        st.caption("Pick a colorway above to see its release date and retail price.")
    elif cw["source"]:
        st.caption(f"Verified at: {cw['source']}")

    section("Shop", "Where to cop")
    links = catalog.buy_links(model_key, colorway_key)
    for col, (site, url) in zip(st.columns(len(links)), links.items()):
        col.link_button(f"{site} ↗", url)


def page_scan():
    hero("AI sneaker identification", 'Snap it.<br><span class="accent">ID it.</span><br>Cop it.',
         "Show SoleSearch any sneaker. The AI names the model and colorway, shows where to buy it, "
         "and saves it to your closet.",
         steps=["Snap", "Identify", "Save"])

    if not MODEL_PATH.exists():
        st.markdown('<div class="ss-empty"><div class="ss-kicker">Offline</div><div class="t">The AI is still training</div>'
                    '<div class="d">No trained model found. Upload <b>solesearch_effnet_b0.pt</b> to the repo and refresh.</div></div>',
                    unsafe_allow_html=True)
        return
    model = load_model()
    matcher = load_matcher(model)

    upload_label, camera_label = "Upload photo", "Use camera"
    if hasattr(st, "segmented_control"):
        source = st.segmented_control("Photo source", [upload_label, camera_label],
                                      default=upload_label, label_visibility="collapsed") or upload_label
    else:
        source = st.radio("Photo source", [upload_label, camera_label], horizontal=True,
                          label_visibility="collapsed")
    if source == upload_label:
        file = st.file_uploader("Drop a sneaker photo", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Point the camera at one sneaker")

    if file is None:
        section("In the lineup", "Sneakers the AI knows")
        sneaker_feed(model.class_names)
        section("Pro tips", "Get the best scan")
        tips = [("One shoe", "A single sneaker per photo works best."),
                ("Side view", "The profile shows the most detail."),
                ("Good light", "Daylight beats a dark closet."),
                ("Fill the frame", "Get close so the shoe is the star.")]
        st.markdown('<div class="ss-tips">' + "".join(
            f'<div class="ss-tip" style="animation-delay:{i * 70}ms"><div class="n">0{i + 1}</div>'
            f'<div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>'
            for i, (t, d) in enumerate(tips)) + "</div>", unsafe_allow_html=True)
        return

    data = file.getvalue()
    photo_id = hashlib.md5(data).hexdigest()[:12]  # resets the choices below when the photo changes
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")  # fixes sideways phone photos

    probs, emb = model.analyze(img)
    top3 = model.top_k(probs, 3)
    best_key, best_p = top3[0]
    confident = best_p >= model.threshold
    label, fg, bg = confidence_badge(best_p, model.threshold)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown(f'<div class="ss-scan"><img src="data:image/jpeg;base64,{pil_b64(img)}" alt="Your photo">'
                    f'<div class="sweep"></div><span class="c tl"></span><span class="c tr"></span>'
                    f'<span class="c bl"></span><span class="c br"></span>'
                    f'<div class="tag"><i></i>Scan complete · {best_p:.0%} match</div></div>',
                    unsafe_allow_html=True)
    with right:
        best = catalog.model(best_key)
        headline = esc(best["name"]) if confident else "Tough one"
        sub = (f'{esc(best["brand"])} · {best_p:.0%} confidence' if confident
               else f'Top guess is only {best_p:.0%}. Here are the closest matches.')
        bars = "".join(
            f'<div class="ss-bar-row"><div class="ss-bar-label"><span>{esc(model_name(k))}</span>'
            f'<span>{p:.1%}</span></div><div class="ss-bar{"" if i == 0 else " dim"}">'
            f'<div style="width:{max(p * 100, 1.5):.1f}%;animation-delay:{200 + i * 120}ms"></div></div></div>'
            for i, (k, p) in enumerate(top3))
        st.markdown(
            f'<div class="ss-card glow"><span class="ss-badge" style="color:{fg};background:{bg}">{label}</span>'
            f'<div class="ss-name">{headline}</div><p class="ss-sub">{sub}</p>'
            f'<div style="margin-top:26px" class="ss-label">Top 3 guesses</div>{bars}</div>',
            unsafe_allow_html=True)

        options = [k for k, _ in top3] + [OTHER]
        choice = st.radio("Confirm the sneaker", options, key=f"model_{photo_id}",
                          format_func=lambda k: "Something else..." if k == OTHER else model_name(k))
        if choice == OTHER:
            choice = st.selectbox("Pick the model", model.class_names, format_func=model_name,
                                  key=f"model_other_{photo_id}")

    if not confident:
        section("Closest matches", "Looks similar to")
        similar = matcher.match(img, emb, model_key=None, k=6)
        if similar:
            for col, m in zip(st.columns(len(similar)), similar):
                col.image(m["photo"])
                col.caption(f"**{model_name(m['model'])}**  \n"
                            f"{catalog.colorway(m['model'], m['colorway'])['name']}")
        else:
            sneaker_feed([k for k, _ in top3])

    section("Colorway", "Which pair is it?")
    matches = matcher.match(img, emb, model_key=choice, k=3)
    if matches:
        for col, m in zip(st.columns(3), matches):
            with col.container(border=True):
                st.image(m["photo"])
                st.markdown(f"**{catalog.colorway(choice, m['colorway'])['name']}**")
                st.caption(f"{m['score']:.0%} match")

    cw_options = []
    for key in ([m["colorway"] for m in matches] + matcher.colorways_for(choice)
                + catalog.colorway_keys(choice)):
        if key not in cw_options:
            cw_options.append(key)
    cw_options.append(UNKNOWN)
    cw_choice = st.selectbox(
        "Colorway", cw_options, key=f"cw_{photo_id}_{choice}",
        format_func=lambda k: "I don't know / not listed" if k == UNKNOWN
        else catalog.colorway(choice, k)["name"])
    colorway_key = None if cw_choice == UNKNOWN else cw_choice

    section("The details", "Know your pair")
    details_section(choice, colorway_key)

    section("Your closet", "Add this pair")
    with st.form(f"add_{photo_id}"):
        c1, c2 = st.columns(2)
        size = c1.text_input("Size (US)", placeholder="e.g. 10.5")
        condition = c2.selectbox("Condition", CONDITIONS)
        price = c1.number_input("What you paid ($)", min_value=0.0, step=5.0, value=None)
        bought = c2.date_input("Date acquired", value=None)
        notes = st.text_area("Notes", placeholder="Gift, where you copped them, etc.")
        submitted = st.form_submit_button("Add to closet")
    if submitted:
        photo_path = closet.save_photo(img)
        closet.add_pair(choice, colorway_key, size or None, condition, price,
                        bought.isoformat() if bought else None, notes or None, photo_path)
        if choice != best_key:  # the AI's first guess was wrong: log it to improve training later
            closet.log_correction(photo_path, best_key, choice, best_p)
        st.toast(f"{model_name(choice)} added to your closet", icon="✅")
        st.success("Secured. Open the **Closet** tab to see it.")


# ---------- page: closet ----------
def page_closet():
    pairs = closet.pairs()
    hero("Your collection", 'My <span class="accent">closet</span>',
         f"{len(pairs)} pair{'s' if len(pairs) != 1 else ''} scanned and saved.", compact=True)
    if not pairs:
        st.markdown('<div class="ss-empty"><div class="ss-kicker">Nothing here yet</div><div class="t">Your closet is empty</div>'
                    '<div class="d">Scan a sneaker and hit <b>Add to closet</b> to start your collection.</div></div>',
                    unsafe_allow_html=True)
        return

    for p in pairs:
        m = catalog.model(p["model_key"])
        cw = catalog.colorway(p["model_key"], p["colorway_key"]) if p["colorway_key"] else None
        p["name"] = m["name"]
        p["brand"] = m["brand"] or "Other"
        p["colorway"] = cw["name"] if cw else "Unknown colorway"
        p["retail"] = cw["retail_price_usd"] if cw else None

    paid = sum(p["purchase_price"] or 0 for p in pairs)
    retail = sum(p["retail"] or 0 for p in pairs)
    most_worn = max(pairs, key=lambda p: p["wear_count"])
    stat_tiles([("Pairs", len(pairs)), ("Total paid", f"${paid:,.0f}"),
                ("Retail value", f"${retail:,.0f}"),
                ("Most worn", most_worn["name"] if most_worn["wear_count"] else "—")])

    section("Breakdown", "By brand")
    st.bar_chart(pd.Series([p["brand"] for p in pairs]).value_counts(), color="#8B5CF6")

    section("The collection", "Every pair")
    f1, f2 = st.columns(2)
    brands = sorted({p["brand"] for p in pairs})
    chosen = f1.multiselect("Filter by brand", brands, default=brands)
    sort = f2.selectbox("Sort by", ["Newest first", "Most worn", "Price paid (high to low)"])
    shown = [p for p in pairs if p["brand"] in chosen]
    if sort == "Most worn":
        shown.sort(key=lambda p: p["wear_count"], reverse=True)
    elif sort.startswith("Price"):
        shown.sort(key=lambda p: p["purchase_price"] or 0, reverse=True)

    cols = st.columns(3)
    for i, p in enumerate(shown):
        with cols[i % 3]:
            with st.container(border=True):
                if p["photo_path"] and Path(p["photo_path"]).exists():
                    st.image(p["photo_path"])
                st.markdown(f'<div class="ss-kicker" style="margin-top:6px">{esc(p["brand"])}</div>'
                            f'<div class="ss-tile-name" style="font-family:Inter Tight,Inter,sans-serif;font-weight:800;font-size:1.25rem;'
                            f'text-transform:uppercase;color:#FFF;letter-spacing:-.02em;margin:4px 0 2px">{esc(p["name"])}</div>'
                            f'<div style="color:#8E8A99;font-size:.9rem">{esc(p["colorway"])}</div>',
                            unsafe_allow_html=True)
                details = [f"Size {p['size']}" if p["size"] else None, p["condition"],
                           f"Paid ${p['purchase_price']:,.0f}" if p["purchase_price"] is not None else None,
                           f"Retail {fmt_price(p['retail'])}" if p["retail"] is not None else None,
                           f"Worn {p['wear_count']}x"]
                st.caption(" · ".join(d for d in details if d))
                if p["notes"]:
                    st.caption(p["notes"])
                b1, b2 = st.columns(2)
                if b1.button("+1 wear", key=f"wear_{p['id']}"):
                    closet.add_wear(p["id"])
                    st.toast(f"Logged a wear for {p['name']}", icon="👟")
                    st.rerun()
                confirm_key = f"confirm_{p['id']}"
                if not st.session_state.get(confirm_key):
                    if b2.button("Remove", key=f"del_{p['id']}"):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    if b2.button("Confirm", key=f"del2_{p['id']}", type="primary"):
                        closet.delete_pair(p["id"])
                        st.session_state.pop(confirm_key, None)
                        st.rerun()

    corrections = closet.corrections()
    with st.expander(f"Corrections log · {len(corrections)} photos the AI got wrong"):
        st.write("Each time you pick a different model than the AI's first guess, the photo and the "
                 "correct label are saved here. Add these photos to the training data and retrain "
                 "to make the model smarter.")
        if corrections:
            st.dataframe(pd.DataFrame(corrections)[
                ["created_at", "predicted_model", "confirmed_model", "confidence", "photo_path"]])


# ---------- page: about the model ----------
def page_about():
    hero("Under the hood", 'How it <span class="accent">works</span>',
         "How SoleSearch recognizes sneakers, and how well it does.", compact=True)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        n = m["num_images"]
        stat_tiles([("Accuracy · top 1", f"{m['test_top1_accuracy']:.1%}"),
                    ("Accuracy · top 3", f"{m['test_top3_accuracy']:.1%}"),
                    ("Confidence threshold", f"{m['confidence_threshold']:.0%}"),
                    ("Training images", f"{n['train']:,}")])
        st.caption(f"Tested on {n['test']} photos the model never saw during training.")
        section("In the lineup", "Sneakers the AI knows")
        sneaker_feed(m["class_names"])
    else:
        st.info("Upload `metrics.json` to the repo to show test results here.")

    section("The pipeline", "Four steps")
    steps = [
        ("Recognize", "EfficientNet-B0, pretrained on ImageNet and fine-tuned on our sneaker dataset, "
                      "predicts the model with its top 3 guesses."),
        ("Stay honest", "Below the confidence threshold, the app says it isn't sure and shows the "
                        "closest matches instead of guessing."),
        ("Match colorway", "Deep features plus a color histogram are compared against reference "
                           "photos of known colorways."),
        ("Learn", "Users confirm or correct each result, and corrections are logged as new training data."),
    ]
    st.markdown('<div class="ss-tips">' + "".join(
        f'<div class="ss-tip" style="animation-delay:{i * 70}ms"><div class="n">0{i + 1}</div>'
        f'<div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>'
        for i, (t, d) in enumerate(steps)) + "</div>", unsafe_allow_html=True)

    charts = [(f, c) for f, c in [("confusion_matrix.png", "Confusion matrix: which sneakers get mixed up"),
                                  ("training_curves.png", "Training curves")]
              if (MODELS_DIR / f).exists()]
    if charts:
        section("Results", "The numbers")
        for col, (fname, caption) in zip(st.columns(len(charts)), charts):
            with col.container(border=True):
                st.image(str(MODELS_DIR / fname), caption=caption)

    if MODEL_PATH.exists():
        section("Maintenance", "Colorway index")
        if st.button("Rebuild colorway index"):
            load_matcher.clear()
            st.success("Reference photos will be re-indexed on the next scan.")
        st.caption("Click after adding photos to the gallery folder.")


# ---------- navigation: brand bar on top, tab bar fixed to the bottom ----------
PAGES = [
    ("scan", "Scan", ":material/photo_camera:"),
    ("closet", "Closet", ":material/checkroom:"),
    ("about", "About", ":material/insights:"),
]
if "page" not in st.session_state:
    st.session_state.page = "scan"


def go_to(page_key):
    st.session_state.page = page_key


pair_count = len(closet.pairs())
st.markdown(f'<div class="ss-topbar"><div class="ss-logo">SOLE<span>SEARCH</span></div>'
            f'<div class="ss-pill"><b>{pair_count}</b> in closet</div></div>', unsafe_allow_html=True)

try:
    nav = st.container(key="bottomnav")  # the key gives it a CSS class, so it can be pinned to the bottom
except TypeError:
    nav = st.container()  # older Streamlit: the tabs just show at the top instead
with nav:
    for col, (page_key, label, icon) in zip(st.columns(len(PAGES)), PAGES):
        col.button(label, icon=icon, key=f"nav_{page_key}", on_click=go_to, args=(page_key,),
                   type="primary" if st.session_state.page == page_key else "secondary")

if st.session_state.page == "scan":
    page_scan()
elif st.session_state.page == "closet":
    page_closet()
else:
    page_about()
