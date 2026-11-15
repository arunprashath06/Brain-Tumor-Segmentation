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
| Browse `archive/kaggle_3m` patients with Dice/IoU | no — the 1.1 GB dataset is not in the repo |
| Training results and saved figures | yes |

The dataset tabs read from `archive/kaggle_3m`; if that folder is absent the app says so instead of
crashing. To demo ground-truth comparison online, either commit a few sample slices or run the app
locally where the dataset lives.

## Alternative: Hugging Face Spaces

Create a **Streamlit** space, push this repo to it (`git remote add space https://huggingface.co/spaces/<user>/<space>`),
and add the same checkpoint via `git lfs`. Spaces gives 16 GB RAM and works with the same `app.py`.
