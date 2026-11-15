# Run the demo online

The repo is deploy-ready for **Streamlit Community Cloud** (free, no server needed) — once the
checkpoint is in the repo, the app runs from a public URL you can paste anywhere.

## The one thing that must be added: the weights

`.pt` files are excluded from git because they are 79–94 MB each. The app needs **one** of them to
make predictions:

```
improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/best_fold0.pt
```

Add it with Git LFS (recommended — keeps the repo light):

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add "improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/best_fold0.pt"
git commit -m "Track the ResNet-34 fold-0 checkpoint with Git LFS"
git push
```

Without the weights the app still starts and shows the *Training results* and *Notes & caveats*
tabs, but the model selector will report that no checkpoint was found.

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → *Deploy from existing repo*.
3. Repository: `arunprashath06/Brain-Tumor-Segmentation` · Branch: `main` · Main file: `app.py`.
4. **Deploy**. First build takes ~5 minutes (PyTorch is large); later pushes redeploy automatically.
5. Copy the public URL — that is the link to submit.

`.streamlit/config.toml` sets the theme and disables usage stats; `requirements.txt` installs the
Python dependencies. OpenCV is `opencv-python-headless`, so no system packages are needed.

## What works online

| Feature | Online |
|---|---|
| Upload an MRI slice and get a mask + confidence | yes (needs the checkpoint) |
| Browse patients with Dice/IoU against ground truth | yes — 3 bundled patients (`archive/demo`) |
| Browse all 110 patients | no — the full 1.1 GB dataset stays local |
| Training results and saved figures | yes |

### The bundled demo subset

The full dataset is 1.1 GB and GitHub rejects anything near that, so `archive/demo/kaggle_3m/`
ships **three complete patients** (17 MB, image and mask for every slice). The hosted app therefore
shows the Dataset tab with real ground truth for those three, and the Upload tab works on any image.
Locally, when `archive/kaggle_3m/` exists, the app prefers it and lists all 110 patients.

To change which patients are bundled:

```bash
mkdir -p archive/demo/kaggle_3m
cp -r "archive/kaggle_3m/TCGA_CS_4941_19960909" archive/demo/kaggle_3m/
git add archive/demo && git commit -m "Bundle another demo patient" && git push
```

Each patient costs roughly 6 MB (23 slices, image plus mask).

## Alternative: Hugging Face Spaces

Create a **Streamlit** space, push this repo to it (`git remote add space https://huggingface.co/spaces/<user>/<space>`),
and add the same checkpoint via `git lfs`. Spaces gives 16 GB RAM and works with the same `app.py`.
