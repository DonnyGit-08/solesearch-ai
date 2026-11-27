"""SoleSearch AI: identify sneakers from a photo and keep a digital closet.

Run with:  streamlit run app.py
"""
import hashlib
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


# ---------- shared resources (loaded once, then reused across reruns) ----------
@st.cache_resource(show_spinner="Loading the AI model...")
def load_model():
    from model import SneakerModel  # imported here so the closet works even without torch
    return SneakerModel(MODEL_PATH)


@st.cache_resource(show_spinner="Indexing colorway reference photos (first run only)...")
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


def fmt_date(d):
    return d if d else "Not verified"


def fmt_price(p):
    return f"${p:,.0f}" if p is not None else "Not verified"


# ---------- page: scan ----------
def info_card(model_key, colorway_key):
    m = catalog.model(model_key)
    cw = catalog.colorway(model_key, colorway_key) if colorway_key else None
    title = m["name"] + (f' "{cw["name"]}"' if cw else "")
    st.markdown(f"### {title}")
    if m["description"]:
        st.write(m["description"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Brand", m["brand"] or "—")
    c2.metric("Release date", fmt_date(cw["release_date"]) if cw else "Pick a colorway")
    c3.metric("Retail price", fmt_price(cw["retail_price_usd"]) if cw else "Pick a colorway")
    if cw and cw["sku"]:
        st.caption(f"SKU: {cw['sku']}")
    if cw and cw["source"]:
        st.caption(f"Verified at: {cw['source']}")

    st.markdown("**Where to buy**")
    links = catalog.buy_links(model_key, colorway_key)
    for col, (site, url) in zip(st.columns(len(links)), links.items()):
        col.link_button(site, url)


def page_scan():
    st.title("👟 SoleSearch AI")
    st.write("Upload or snap a photo of a sneaker. The AI identifies the model and colorway, "
             "then you can save it to your closet.")

    if not MODEL_PATH.exists():
        st.error(f"No trained model found. Download `solesearch_effnet_b0.pt` from "
                 f"`SoleSearch/models/` in Google Drive and put it in:\n\n`{MODELS_DIR}`")
        return
    model = load_model()
    matcher = load_matcher(model)

    source = st.radio("Photo source", ["Upload a photo", "Take a photo"], horizontal=True)
    if source == "Upload a photo":
        file = st.file_uploader("Sneaker photo", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Point the camera at one sneaker")
    if file is None:
        st.info("Tips: one sneaker per photo, side view works best, decent lighting, "
                "and fill most of the frame with the shoe.")
        return

    data = file.getvalue()
    photo_id = hashlib.md5(data).hexdigest()[:12]  # resets the choices below when the photo changes
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")  # fixes sideways phone photos

    probs, emb = model.analyze(img)
    top3 = model.top_k(probs, 3)
    best_key, best_p = top3[0]
    confident = best_p >= model.threshold

    left, right = st.columns([1, 1.3])
    with left:
        st.image(img)
    with right:
        if confident:
            st.subheader(f"Looks like: {model_name(best_key)}")
            st.caption(f"{best_p:.0%} confident")
        else:
            st.subheader("Not sure, here are the closest matches")
            st.caption(f"Top confidence is {best_p:.0%}, below the {model.threshold:.0%} "
                       "the app needs to give a direct answer.")
        st.markdown("**AI's top 3 guesses**")
        for key, p in top3:
            st.progress(p, text=f"{model_name(key)}: {p:.1%}")

        options = [k for k, _ in top3] + [OTHER]
        choice = st.radio("Which sneaker is it?", options, key=f"model_{photo_id}",
                          format_func=lambda k: "Something else..." if k == OTHER else model_name(k))
        if choice == OTHER:
            choice = st.selectbox("Pick the model", model.class_names, format_func=model_name,
                                  key=f"model_other_{photo_id}")

    if not confident:
        st.markdown("#### Visually similar sneakers")
        similar = matcher.match(img, emb, model_key=None, k=6)
        if similar:
            for col, m in zip(st.columns(len(similar)), similar):
                col.image(m["photo"], width=140)
                col.caption(f"{model_name(m['model'])}\n\n"
                            f"{catalog.colorway(m['model'], m['colorway'])['name']}")
        else:
            st.caption("Add reference photos to the gallery folder to see similar sneakers.")

    st.divider()
    st.markdown("#### Colorway")
    matches = matcher.match(img, emb, model_key=choice, k=3)
    if matches:
        for col, m in zip(st.columns(3), matches):
            col.image(m["photo"], width=160)
            col.markdown(f"**{catalog.colorway(choice, m['colorway'])['name']}**")
            col.caption(f"Similarity {m['score']:.0%}")
    else:
        st.info(f"No reference photos for {model_name(choice)} yet. Add 3–5 photos per colorway to "
                f"`gallery/{choice}/<colorway>/` and click **Rebuild colorway index** in the sidebar.")

    cw_options = []
    for key in ([m["colorway"] for m in matches] + matcher.colorways_for(choice)
                + catalog.colorway_keys(choice)):
        if key not in cw_options:
            cw_options.append(key)
    cw_options.append(UNKNOWN)
    cw_choice = st.selectbox(
        "Confirm the colorway", cw_options, key=f"cw_{photo_id}_{choice}",
        format_func=lambda k: "I don't know / not listed" if k == UNKNOWN
        else catalog.colorway(choice, k)["name"])
    colorway_key = None if cw_choice == UNKNOWN else cw_choice

    st.divider()
    info_card(choice, colorway_key)

    st.divider()
    st.markdown("#### Add to my closet")
    with st.form(f"add_{photo_id}"):
        c1, c2 = st.columns(2)
        size = c1.text_input("Size (US)", placeholder="e.g. 10.5")
        condition = c2.selectbox("Condition", CONDITIONS)
        price = c1.number_input("What you paid ($)", min_value=0.0, step=5.0, value=None)
        bought = c2.date_input("Date acquired", value=None)
        notes = st.text_area("Notes", placeholder="Gift, where you bought them, etc.")
        submitted = st.form_submit_button("Add to closet")
    if submitted:
        photo_path = closet.save_photo(img)
        closet.add_pair(choice, colorway_key, size or None, condition, price,
                        bought.isoformat() if bought else None, notes or None, photo_path)
        if choice != best_key:  # the AI's first guess was wrong: log it to improve training later
            closet.log_correction(photo_path, best_key, choice, best_p)
        st.success(f"Added {model_name(choice)} to your closet!")


# ---------- page: closet ----------
def page_closet():
    st.title("🗄️ My Closet")
    pairs = closet.pairs()
    if not pairs:
        st.info("Your closet is empty. Scan a sneaker and click **Add to closet**.")
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs", len(pairs))
    c2.metric("Total paid", f"${paid:,.0f}")
    c3.metric("Retail value (verified)", f"${retail:,.0f}")
    c4.metric("Most worn", most_worn["name"] if most_worn["wear_count"] else "—")

    st.markdown("**Pairs by brand**")
    st.bar_chart(pd.Series([p["brand"] for p in pairs]).value_counts())

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
                st.markdown(f"**{p['name']}**  \n{p['colorway']}")
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
                    st.rerun()
                confirm_key = f"confirm_{p['id']}"
                if not st.session_state.get(confirm_key):
                    if b2.button("Remove", key=f"del_{p['id']}"):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    if b2.button("Confirm remove", key=f"del2_{p['id']}", type="primary"):
                        closet.delete_pair(p["id"])
                        st.session_state.pop(confirm_key, None)
                        st.rerun()

    corrections = closet.corrections()
    with st.expander(f"Corrections log ({len(corrections)} photos the AI got wrong)"):
        st.write("Each time you pick a different model than the AI's first guess, the photo and the "
                 "correct label are saved here. Copy these photos into `train_data/<model>/` "
                 "and retrain to make the model better.")
        if corrections:
            st.dataframe(pd.DataFrame(corrections)[
                ["created_at", "predicted_model", "confirmed_model", "confidence", "photo_path"]])


# ---------- page: about the model ----------
def page_about():
    st.title("📊 About the model")
    st.markdown("""
**How SoleSearch works**
1. **Model recognition:** an EfficientNet-B0 network, pretrained on ImageNet and fine-tuned on our
   sneaker dataset, predicts the sneaker model and gives its top 3 guesses with confidence.
2. **Honest uncertainty:** below a confidence threshold, the app says it isn't sure and shows
   visually similar sneakers instead of guessing.
3. **Colorway matching:** the photo's deep-feature embedding and its color histogram are compared
   against reference photos of known colorways.
4. **User confirmation:** the user confirms or corrects the result before it's saved to the closet,
   and corrections are logged as new training data.
""")
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy (top 1)", f"{m['test_top1_accuracy']:.1%}")
        c2.metric("Test accuracy (top 3)", f"{m['test_top3_accuracy']:.1%}")
        c3.metric("Confidence threshold", f"{m['confidence_threshold']:.0%}")
        n = m["num_images"]
        c4.metric("Training images", f"{n['train']:,}", help=f"Validation {n['val']}, test {n['test']}")
        st.caption("Sneaker classes: " + ", ".join(model_name(c) for c in m["class_names"]))
    else:
        st.info("Copy `metrics.json` from `SoleSearch/models/` in Drive into the `models` folder "
                "to show test results here.")
    for fname, caption in [("confusion_matrix.png", "Confusion matrix (test set)"),
                           ("training_curves.png", "Training curves")]:
        if (MODELS_DIR / fname).exists():
            st.image(str(MODELS_DIR / fname), caption=caption)


# ---------- sidebar navigation ----------
st.sidebar.title("SoleSearch AI")
page = st.sidebar.radio("Go to", ["Scan a sneaker", "My Closet", "About the model"])
st.sidebar.divider()
if MODEL_PATH.exists():
    if st.sidebar.button("Rebuild colorway index"):
        load_matcher.clear()
        st.sidebar.success("Reference photos will be re-indexed.")
    st.sidebar.caption("Click after adding photos to the gallery folder.")

if page == "Scan a sneaker":
    page_scan()
elif page == "My Closet":
    page_closet()
else:
    page_about()
