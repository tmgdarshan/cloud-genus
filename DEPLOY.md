# Deploying Cloud Genus (PWA)

The app is plain static files. It needs **HTTPS** (for camera, GPS, and "Add to Home
Screen") — so a local `http.server` won't do for the phone; it has to be hosted.

## Option A — GitHub Pages (recommended)

Permanent link, free, no per-file size limit issues (the models are ~23 MB each).

```bash
cd D:\cloud-classification-cnn-mobile\app

git init -b main
git add -A
git commit -m "Cloud Genus PWA"

# create an EMPTY public repo named cloud-genus on github.com first, then:
git remote add origin https://github.com/tmgdarshan/cloud-genus.git
git push -u origin main
```

Then on github.com: **repo → Settings → Pages → Source: Deploy from a branch → `main` / `/ (root)` → Save.**

Live in ~1 min at **https://tmgdarshan.github.io/cloud-genus/**

Open that on your phone → browser menu → **Add to Home Screen**. It installs like an app,
runs full-screen, and works offline after the first load.

To update later: `git add -A && git commit -m "…" && git push` — Pages redeploys automatically.

## Option B — Cloudflare Pages (direct upload)

```bash
cd D:\cloud-classification-cnn-mobile\app
npx wrangler pages deploy . --project-name cloud-genus
```

Opens a browser once to log in. Gives `https://cloud-genus.pages.dev`.
Note: Cloudflare's free tier caps files at 25 MB — the fp16 models (22.6 MB) just fit.

## Option C — quick test without deploying

Tunnel the local server straight to your phone (no hosting, PC must stay on):

```bash
# terminal 1
cd D:\cloud-classification-cnn-mobile\app && python -m http.server 8080
# terminal 2  (download cloudflared.exe from github.com/cloudflare/cloudflared/releases)
cloudflared tunnel --url http://localhost:8080
```

Open the printed `https://….trycloudflare.com` link on your phone.

## Files that ship

Everything in `app/` except `tools/` and the `*.md` files is the app. `.nojekyll` tells
GitHub Pages to serve files as-is. The model, app shell, and onnxruntime are cached by
`sw.js` for offline use.
