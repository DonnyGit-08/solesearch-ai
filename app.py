"""SoleSearch AI: identify sneakers from a photo and keep a digital closet.

Run with:  streamlit run app.py
"""
import hashlib
import html
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

# ---------- look and feel ----------
# Custom blocks set their own text and background colors, so they stay readable
# whether the viewer's Streamlit theme is light or dark.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.03em; }
footer { visibility: hidden; }

@keyframes ssUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes ssGrow { from { width: 0; } }
@keyframes ssFloat { 0%,100% { transform: translateY(-50%) rotate(-14deg); } 50% { transform: translateY(-56%) rotate(-10deg); } }

/* hero banner */
.ss-hero { position: relative; overflow: hidden; border-radius: 22px; padding: 36px 34px 30px;
  background: radial-gradient(circle at 85% 20%, rgba(124,58,237,.55), transparent 45%),
              linear-gradient(135deg, #0F0D14 0%, #1C1430 100%);
  color: #F7F5FB; animation: ssUp .5s ease; margin-bottom: 18px; }
.ss-hero h1 { color: #FFFFFF !important; font-size: 2.8rem; line-height: 1.05; margin: 0 0 8px; padding: 0; }
.ss-hero h1 .accent { color: #C4A7FF; }
.ss-hero p { color: #CDC6DC; font-size: 1.05rem; max-width: 560px; margin: 0; }
.ss-hero.compact { padding: 24px 28px; }
.ss-hero.compact h1 { font-size: 2rem; }
.ss-hero .ss-emoji { position: absolute; right: 28px; top: 50%; font-size: 6rem; opacity: .95;
  animation: ssFloat 4s ease-in-out infinite; filter: drop-shadow(0 12px 18px rgba(0,0,0,.4)); }
.ss-steps { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
.ss-step { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
  border-radius: 999px; padding: 6px 14px; font-size: .9rem; color: #F7F5FB; }
.ss-step b { color: #C4A7FF; margin-right: 4px; }
@media (max-width: 700px) { .ss-hero .ss-emoji { display: none; } .ss-hero h1 { font-size: 1.9rem; } }

/* cards */
.ss-card { background: #FFFFFF; color: #0F0D14; border: 1px solid #ECE8F3; border-radius: 20px;
  padding: 22px 24px; box-shadow: 0 1px 2px rgba(15,13,20,.04), 0 10px 30px rgba(15,13,20,.05); animation: ssUp .45s ease; margin-bottom: 14px; }
.ss-card.warm { background: #F7F5FB; border-color: #E6E0F2; }
.ss-badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700;
  font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
.ss-kicker { font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #8B86A0; }
.ss-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.03em; font-size: 2.1rem; line-height: 1.08; margin: 10px 0 4px; color: #0F0D14; }
.ss-sub { color: #6B6680; font-size: .98rem; margin: 0; }
.ss-desc { color: #3A3548; margin: 12px 0 0; line-height: 1.5; }

/* confidence bars */
.ss-bar-row { margin: 12px 0 0; }
.ss-bar-label { display: flex; justify-content: space-between; font-weight: 600; font-size: .92rem; color: #0F0D14; }
.ss-bar-label span:last-child { color: #6B6680; font-variant-numeric: tabular-nums; }
.ss-bar { height: 10px; background: #EEEAF5; border-radius: 999px; overflow: hidden; margin-top: 5px; }
.ss-bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #A78BFA, #7C3AED);
  animation: ssGrow .9s cubic-bezier(.2,.8,.2,1); }
.ss-bar.dim > div { background: #D6D0E4; }

/* stat tiles */
.ss-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 4px; }
.ss-stat { background: #F7F5FB; border: 1px solid #ECE8F3; border-radius: 16px; padding: 14px 16px; color: #0F0D14; animation: ssUp .45s ease; }
.ss-stat .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: #7C7790; font-weight: 700; }
.ss-stat .v { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.03em; font-size: 1.45rem; margin-top: 4px; }
.ss-stat .v.muted { font-family: 'Inter', sans-serif; font-size: 1rem; color: #A9A4BC; font-weight: 600; }

/* chips */
.ss-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.ss-chip { background: #0F0D14; color: #F7F5FB; border-radius: 999px; padding: 6px 12px; font-size: .85rem; font-weight: 600; }

/* tip tiles */
.ss-tips { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.ss-tip { background: #FFFFFF; border: 1px solid #ECE8F3; border-radius: 18px; padding: 18px; color: #0F0D14; animation: ssUp .5s ease; }
.ss-tip .e { font-size: 1.8rem; }
.ss-tip .t { font-weight: 700; margin-top: 6px; }
.ss-tip .d { color: #6B6680; font-size: .9rem; margin-top: 2px; }

/* empty state */
.ss-empty { text-align: center; padding: 48px 24px; background: #F7F5FB; border: 2px dashed #D9CCF5;
  border-radius: 24px; color: #0F0D14; animation: ssUp .5s ease; }
.ss-empty .e { font-size: 3.5rem; }
.ss-empty .t { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.03em; font-size: 1.4rem; margin: 8px 0 4px; }
.ss-empty .d { color: #6B6680; }

/* sidebar brand */
.ss-logo { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.03em; font-size: 1.5rem; letter-spacing: -.02em; }
.ss-logo span { color: #7C3AED; }
.ss-tag { font-size: .85rem; opacity: .7; margin-top: -4px; }

/* Streamlit widgets */
[data-testid="stImage"] img { border-radius: 16px; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #7C3AED !important; border-radius: 20px !important;
  background: rgba(124,58,237,.05) !important; padding: 28px !important; }
div.stButton > button, [data-testid="stLinkButton"] a { border-radius: 999px !important; font-weight: 600 !important;
  transition: transform .12s ease, box-shadow .12s ease; }
div.stButton > button:hover, [data-testid="stLinkButton"] a:hover { transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(124,58,237,.25); }
[data-testid="stFormSubmitButton"] button { background: #7C3AED !important; color: #FFFFFF !important; border: none !important;
  border-radius: 999px !important; font-weight: 700 !important; padding: .6rem 1.6rem !important; }
[data-testid="stFormSubmitButton"] button:hover { background: #6D28D9 !important; }
[data-testid="stForm"] { border-radius: 20px !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px !important; transition: transform .15s ease, box-shadow .15s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: 0 10px 26px rgba(15,13,20,.09); }

/* top brand bar (the sidebar is replaced by the bottom tab bar) */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.ss-topbar { display: flex; align-items: baseline; gap: 12px; margin: -6px 0 14px; color: #0F0D14; }
.ss-topbar .ss-tag { margin: 0; }

/* placeholder when a sneaker has no display photo yet */
.ss-ph { aspect-ratio: 4 / 3; display: flex; align-items: center; justify-content: center; font-size: 2.6rem;
  background: linear-gradient(135deg, #F3F1F8, #E9E5F2); border-radius: 14px; }

/* bottom tab bar */
[data-testid="stMainBlockContainer"], .block-container { padding-bottom: 130px !important; }
.st-key-bottomnav { position: fixed; left: 50%; transform: translateX(-50%); bottom: 18px; z-index: 1000;
  width: min(480px, calc(100% - 32px)); padding: 8px; border-radius: 24px;
  background: rgba(15,13,20,.92); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 14px 40px rgba(15,13,20,.35); }
.st-key-bottomnav [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 6px !important; }
.st-key-bottomnav [data-testid="stColumn"], .st-key-bottomnav [data-testid="column"] {
  min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
.st-key-bottomnav button { width: 100% !important; border: none !important; border-radius: 18px !important;
  background: transparent !important; color: #B9B2CC !important; padding: 10px 4px !important;
  box-shadow: none !important; transform: none !important; }
.st-key-bottomnav button:hover { color: #FFFFFF !important; background: rgba(255,255,255,.07) !important; }
.st-key-bottomnav button[kind="primary"], .st-key-bottomnav button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #8B5CF6, #7C3AED) !important; color: #FFFFFF !important;
  box-shadow: 0 6px 18px rgba(124,58,237,.45) !important; }
.st-key-bottomnav button p { font-size: .88rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def hero(title_html, subtitle, emoji="👟", steps=None, compact=False):
    steps_html = ""
    if steps:
        steps_html = '<div class="ss-steps">' + "".join(
            f'<div class="ss-step"><b>{i}</b>{esc(s)}</div>' for i, s in enumerate(steps, 1)) + "</div>"
    st.markdown(
        f'<div class="ss-hero{" compact" if compact else ""}"><div class="ss-emoji">{emoji}</div>'
        f'<h1>{title_html}</h1><p>{esc(subtitle)}</p>{steps_html}</div>',
        unsafe_allow_html=True)


def stat_tiles(items):
    """items: list of (label, value). None values show as 'Not verified'."""
    tiles = []
    for label, value in items:
        if value is None:
            tiles.append(f'<div class="ss-stat"><div class="k">{esc(label)}</div><div class="v muted">Not verified</div></div>')
        else:
            tiles.append(f'<div class="ss-stat"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div></div>')
    st.markdown(f'<div class="ss-stats">{"".join(tiles)}</div>', unsafe_allow_html=True)


def confidence_badge(p, threshold):
    if p >= 0.85:
        return "🔥 Locked in", "#166534", "#DCFCE7"
    if p >= threshold:
        return "👍 Pretty sure", "#92400E", "#FEF3C7"
    return "🤔 Not sure", "#991B1B", "#FEE2E2"


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


def display_image(key):
    """A real photo for each sneaker model: displays/<key>.jpg or <key>.jpg at the top of the repo."""
    for folder in (ROOT / "displays", ROOT):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = folder / f"{key}{ext}"
            if path.exists():
                return str(path)
    return None


def known_sneakers(class_names, note=None):
    st.markdown("### 👟 Sneakers the AI knows")
    if note:
        st.caption(note)
    cols = st.columns(4)
    for i, key in enumerate(class_names):
        with cols[i % 4].container(border=True):
            img = display_image(key)
            if img:
                st.image(img)
            else:
                st.markdown('<div class="ss-ph">👟</div>', unsafe_allow_html=True)
            st.markdown(f"**{model_name(key)}**")
            st.caption(catalog.model(key)["brand"] or " ")


def fmt_price(p):
    return f"${p:,.0f}" if p is not None else None


# ---------- page: scan ----------
def details_section(model_key, colorway_key):
    m = catalog.model(model_key)
    cw = catalog.colorway(model_key, colorway_key) if colorway_key else None
    title = m["name"] + (f' “{cw["name"]}”' if cw else "")
    desc = f'<p class="ss-desc">{esc(m["description"])}</p>' if m["description"] else ""
    card = (f'<div class="ss-card warm"><div class="ss-kicker">The details</div>'
            f'<div class="ss-name">{esc(title)}</div>{desc}</div>')
    img = display_image(model_key)
    if img:
        c_img, c_text = st.columns([1, 2], gap="medium")
        c_img.image(img)
        c_text.markdown(card, unsafe_allow_html=True)
    else:
        st.markdown(card, unsafe_allow_html=True)
    items = [("Brand", m["brand"] or "—")]
    if cw:
        items += [("Release date", cw["release_date"]), ("Retail price", fmt_price(cw["retail_price_usd"]))]
        if cw["sku"]:
            items.append(("SKU", cw["sku"]))
    stat_tiles(items)
    if not cw:
        st.caption("Pick a colorway above to see its release date and retail price.")
    elif cw["source"]:
        st.caption(f"Verified at: {cw['source']}")

    st.markdown("### 🛒 Where to cop")
    links = catalog.buy_links(model_key, colorway_key)
    for col, (site, url) in zip(st.columns(len(links)), links.items()):
        col.link_button(f"{site} ↗", url)


def page_scan():
    hero('Snap it. <span class="accent">ID it.</span> Cop it.',
         "Show SoleSearch a sneaker and the AI tells you the model, the colorway, "
         "and where to buy it. Then flex it in your closet.",
         steps=["Snap a photo", "AI identifies it", "Save to your closet"])

    if not MODEL_PATH.exists():
        st.markdown('<div class="ss-empty"><div class="e">🧠</div><div class="t">The AI is still in the gym</div>'
                    '<div class="d">No trained model found. Upload <b>solesearch_effnet_b0.pt</b> to the repo '
                    'and refresh.</div></div>', unsafe_allow_html=True)
        return
    model = load_model()
    matcher = load_matcher(model)

    upload_label, camera_label = "📁  Upload a photo", "📸  Use camera"
    if hasattr(st, "segmented_control"):
        source = st.segmented_control("Photo source", [upload_label, camera_label],
                                      default=upload_label, label_visibility="collapsed") or upload_label
    else:
        source = st.radio("Photo source", [upload_label, camera_label], horizontal=True,
                          label_visibility="collapsed")
    if source == upload_label:
        file = st.file_uploader("Drop a sneaker photo here", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Point the camera at one sneaker")

    if file is None:
        st.markdown("""
<div class="ss-tips">
  <div class="ss-tip"><div class="e">👟</div><div class="t">One shoe at a time</div><div class="d">A single sneaker per photo works best.</div></div>
  <div class="ss-tip"><div class="e">📐</div><div class="t">Side view wins</div><div class="d">The profile shows the most detail.</div></div>
  <div class="ss-tip"><div class="e">💡</div><div class="t">Good lighting</div><div class="d">Daylight beats a dark closet.</div></div>
  <div class="ss-tip"><div class="e">🔍</div><div class="t">Fill the frame</div><div class="d">Get close so the shoe is the star.</div></div>
</div>""", unsafe_allow_html=True)
        known_sneakers(model.class_names)
        return

    data = file.getvalue()
    photo_id = hashlib.md5(data).hexdigest()[:12]  # resets the choices below when the photo changes
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")  # fixes sideways phone photos

    with st.spinner("Studying the silhouette..."):
        probs, emb = model.analyze(img)
    top3 = model.top_k(probs, 3)
    best_key, best_p = top3[0]
    confident = best_p >= model.threshold
    label, fg, bg = confidence_badge(best_p, model.threshold)

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(img)
    with right:
        best = catalog.model(best_key)
        headline = esc(best["name"]) if confident else "Hmm, tough one"
        sub = (f'{esc(best["brand"])} · {best_p:.0%} confident' if confident
               else f'Best guess is {best_p:.0%}, so here are the closest matches.')
        bars = "".join(
            f'<div class="ss-bar-row"><div class="ss-bar-label"><span>{esc(model_name(k))}</span>'
            f'<span>{p:.1%}</span></div><div class="ss-bar{"" if i == 0 else " dim"}">'
            f'<div style="width:{max(p * 100, 1.5):.1f}%"></div></div></div>'
            for i, (k, p) in enumerate(top3))
        st.markdown(
            f'<div class="ss-card"><span class="ss-badge" style="color:{fg};background:{bg}">{label}</span>'
            f'<div class="ss-name">{headline}</div><p class="ss-sub">{sub}</p>'
            f'<div style="margin-top:16px" class="ss-kicker">Top 3 guesses</div>{bars}</div>',
            unsafe_allow_html=True)

        options = [k for k, _ in top3] + [OTHER]
        choice = st.radio("Is this right? Pick the correct sneaker:", options, key=f"model_{photo_id}",
                          format_func=lambda k: "Something else..." if k == OTHER else model_name(k))
        if choice == OTHER:
            choice = st.selectbox("Pick the model", model.class_names, format_func=model_name,
                                  key=f"model_other_{photo_id}")

    if not confident:
        st.markdown("### 👀 Looks similar to")
        similar = matcher.match(img, emb, model_key=None, k=6)
        if similar:
            for col, m in zip(st.columns(len(similar)), similar):
                col.image(m["photo"])
                col.caption(f"**{model_name(m['model'])}**  \n"
                            f"{catalog.colorway(m['model'], m['colorway'])['name']}")
        else:
            st.caption("Add reference photos to the gallery folder to see similar sneakers here.")

    st.markdown("### 🎨 Colorway")
    matches = matcher.match(img, emb, model_key=choice, k=3)
    if matches:
        for col, m in zip(st.columns(3), matches):
            with col.container(border=True):
                st.image(m["photo"])
                st.markdown(f"**{catalog.colorway(choice, m['colorway'])['name']}**")
                st.caption(f"{m['score']:.0%} match")
    else:
        st.caption(f"No reference photos for {model_name(choice)} yet, so pick the colorway from the list.")

    cw_options = []
    for key in ([m["colorway"] for m in matches] + matcher.colorways_for(choice)
                + catalog.colorway_keys(choice)):
        if key not in cw_options:
            cw_options.append(key)
    cw_options.append(UNKNOWN)
    cw_choice = st.selectbox(
        "Which colorway is it?", cw_options, key=f"cw_{photo_id}_{choice}",
        format_func=lambda k: "I don't know / not listed" if k == UNKNOWN
        else catalog.colorway(choice, k)["name"])
    colorway_key = None if cw_choice == UNKNOWN else cw_choice

    st.write("")
    details_section(choice, colorway_key)

    st.markdown("### ➕ Add to your closet")
    with st.form(f"add_{photo_id}"):
        c1, c2 = st.columns(2)
        size = c1.text_input("Size (US)", placeholder="e.g. 10.5")
        condition = c2.selectbox("Condition", CONDITIONS)
        price = c1.number_input("What you paid ($)", min_value=0.0, step=5.0, value=None)
        bought = c2.date_input("Date acquired", value=None)
        notes = st.text_area("Notes", placeholder="Gift, where you copped them, etc.")
        submitted = st.form_submit_button("Add to closet  👟")
    if submitted:
        photo_path = closet.save_photo(img)
        closet.add_pair(choice, colorway_key, size or None, condition, price,
                        bought.isoformat() if bought else None, notes or None, photo_path)
        if choice != best_key:  # the AI's first guess was wrong: log it to improve training later
            closet.log_correction(photo_path, best_key, choice, best_p)
        st.balloons()
        st.toast(f"{model_name(choice)} added to your closet!", icon="✅")
        st.success("Saved! Open **My Closet** in the sidebar to see it.")


# ---------- page: closet ----------
def page_closet():
    hero('My <span class="accent">Closet</span>', "Every pair you've scanned and saved, all in one place.",
         emoji="🗄️", compact=True)
    pairs = closet.pairs()
    if not pairs:
        st.markdown('<div class="ss-empty"><div class="e">👟</div><div class="t">Your closet is empty</div>'
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

    st.markdown("### 📊 Brand breakdown")
    st.bar_chart(pd.Series([p["brand"] for p in pairs]).value_counts(), color="#7C3AED")

    st.markdown("### 👟 The collection")
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
                st.markdown(f"#### {p['name']}")
                st.markdown(f"🎨 {p['colorway']}")
                details = [f"Size {p['size']}" if p["size"] else None, p["condition"],
                           f"Paid ${p['purchase_price']:,.0f}" if p["purchase_price"] is not None else None,
                           f"Retail {fmt_price(p['retail'])}" if p["retail"] is not None else None,
                           f"Worn {p['wear_count']}x"]
                st.caption(" · ".join(d for d in details if d))
                if p["notes"]:
                    st.caption(f"📝 {p['notes']}")
                b1, b2 = st.columns(2)
                if b1.button("👟 +1 wear", key=f"wear_{p['id']}"):
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
    with st.expander(f"🛠️ Corrections log ({len(corrections)} photos the AI got wrong)"):
        st.write("Each time you pick a different model than the AI's first guess, the photo and the "
                 "correct label are saved here. Add these photos to the training data and retrain "
                 "to make the model smarter.")
        if corrections:
            st.dataframe(pd.DataFrame(corrections)[
                ["created_at", "predicted_model", "confirmed_model", "confidence", "photo_path"]])


# ---------- page: about the model ----------
def page_about():
    hero('Under the <span class="accent">hood</span>', "How SoleSearch recognizes sneakers, and how well it does.",
         emoji="🧠", compact=True)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        n = m["num_images"]
        stat_tiles([("Accuracy (top 1)", f"{m['test_top1_accuracy']:.1%}"),
                    ("Accuracy (top 3)", f"{m['test_top3_accuracy']:.1%}"),
                    ("Confidence threshold", f"{m['confidence_threshold']:.0%}"),
                    ("Training images", f"{n['train']:,}")])
        known_sneakers(m["class_names"],
                       note=f"Tested on {n['test']} photos the model never saw during training.")
    else:
        st.info("Upload `metrics.json` to the repo to show test results here.")

    st.markdown("### ⚙️ How it works")
    steps = [
        ("🧠", "Recognize", "EfficientNet-B0, pretrained on ImageNet and fine-tuned on our sneaker dataset, "
                           "predicts the model with its top 3 guesses."),
        ("🤔", "Stay honest", "Below the confidence threshold, the app says it isn't sure and shows similar "
                             "sneakers instead of guessing."),
        ("🎨", "Match colorway", "Deep features plus a color histogram are compared against reference "
                                "photos of known colorways."),
        ("✅", "Learn", "Users confirm or correct each result, and corrections are logged as new training data."),
    ]
    tiles = "".join(f'<div class="ss-tip"><div class="e">{e}</div><div class="t">{esc(t)}</div>'
                    f'<div class="d">{esc(d)}</div></div>' for e, t, d in steps)
    st.markdown(f'<div class="ss-tips">{tiles}</div>', unsafe_allow_html=True)

    charts = [(f, c) for f, c in [("confusion_matrix.png", "Confusion matrix: which sneakers get mixed up"),
                                  ("training_curves.png", "Training curves")]
              if (MODELS_DIR / f).exists()]
    if charts:
        st.markdown("### 📈 Results")
        for col, (fname, caption) in zip(st.columns(len(charts)), charts):
            with col.container(border=True):
                st.image(str(MODELS_DIR / fname), caption=caption)

    if MODEL_PATH.exists():
        st.markdown("### 🛠️ Maintenance")
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
            f'<div class="ss-tag">AI sneaker ID · {pair_count} pair{"s" if pair_count != 1 else ""} '
            f'in your closet</div></div>', unsafe_allow_html=True)

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
