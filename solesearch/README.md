# SoleSearch AI

Identify sneakers from a photo (model + colorway), see release info and buy links, and keep a digital closet.

## Folder layout

```
solesearch/
  app.py              the Streamlit app
  catalog.json        sneaker facts (fill in release dates, prices, sources)
  requirements.txt
  core/               model, colorway matcher, catalog, closet database
  models/             put solesearch_effnet_b0.pt here (+ metrics.json and the .png charts)
  gallery/            colorway reference photos: gallery/<model>/<colorway>/*.jpg
  data/               created automatically: closet.db, saved photos, gallery index
```

## Setup (one time)

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)
   source .venv/bin/activate       (Mac)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   The first `pip install` gets the smaller CPU-only PyTorch, which is all the app needs.
3. From Google Drive, `SoleSearch/models/`, download these into `models/`:
   - `solesearch_effnet_b0.pt` (required)
   - `metrics.json`, `confusion_matrix.png`, `training_curves.png` (shown on the About page)
4. Copy your colorway reference photos into `gallery/` (same folder names as in Drive).

## Run

```
streamlit run app.py
```

It opens in your browser at http://localhost:8501.

## Adding sneakers later

- **New colorway:** add a folder `gallery/<model>/<colorway>/` with 3–5 photos, add an entry in `catalog.json`, click **Rebuild colorway index** in the sidebar.
- **New model:** add `train_data/<model>/` in Drive, retrain with notebook 2, replace the `.pt` file, add the model to `catalog.json`.

Folder names in `train_data`, `gallery` and `catalog.json` must match exactly (e.g. `air_jordan_1`).
