# Deploying Cloud Genus (PWA)

The app is plain static files. It needs HTTPS for the camera, GPS, and "Add to
Home Screen", so a local `http.server` will not do for the phone. It has to be
hosted.

## Option A, GitHub Pages (recommended)

Permanent link, free, and no per-file size limit that matters here (the models
are about 23 MB each).

The `app/` folder is its own git repo with its own remote (`cloud-genus`),
separate from the research repo it sits inside. The research repo does not track
`app/`.

Do not run `git init` or `git add` for `app/` from the research repo root. That
would pull this folder and its 68 MB of model files into the research history.
Run every git command for the app from inside `app/`.

First-time setup, already done, kept here for reference:

```bash
cd D:\cloud-classification-cnn-mobile\app
git init -b main
git remote add origin https://github.com/tmgdarshan/cloud-genus.git
git add -A && git commit -m "Cloud Genus PWA"
git push -u origin main
```

Then in the `cloud-genus` repo on github.com open Settings, then Pages, then set
Source to "Deploy from a branch" with branch `main` and folder `/ (root)`, then
Save. The site is live in about a minute at
`https://tmgdarshan.github.io/cloud-genus/`.

To update later:

```bash
cd D:\cloud-classification-cnn-mobile\app
git add -A && git commit -m "message" && git push
```

Open the site on your phone, then the browser menu, then "Add to Home Screen". It
installs like an app, runs full-screen, and works offline after the first load.

## Option B, Cloudflare Pages (direct upload)

```bash
cd D:\cloud-classification-cnn-mobile\app
npx wrangler pages deploy . --project-name cloud-genus
```

Opens a browser once to log in. Gives `https://cloud-genus.pages.dev`. Cloudflare's
free tier caps files at 25 MB, which the fp16 models (22.6 MB) just fit under.

## Option C, quick test without deploying

Tunnel the local server straight to your phone. No hosting, but the PC must stay
on.

```bash
# terminal 1
cd D:\cloud-classification-cnn-mobile\app && python -m http.server 8080
# terminal 2, using cloudflared.exe from github.com/cloudflare/cloudflared/releases
cloudflared tunnel --url http://localhost:8080
```

Open the printed `https://<name>.trycloudflare.com` link on your phone.

## Files that ship

Everything in `app/` except `tools/` and the `.md` files is the app. `.nojekyll`
tells GitHub Pages to serve files as they are. The model, app shell, and
onnxruntime are cached by `sw.js` for offline use.
