# Cloud Genus phone app (PWA)

Installable web app. Take a photo of the sky and get the cloud genus, what it
means, your local weather, and a short plain-language advisory. The classifier
runs on the device (ONNX in the browser). Only the optional weather lookup needs
a connection.

The model is a ResNet-18 trained on the CCSN dataset to name the 11 WMO cloud
genera (Ac, As, Cb, Cc, Ci, Cs, Ct, Cu, Ns, Sc, St). Its hyperparameters come
from the CCSN benchmark in the wider research project.

## Run locally

```
cd app
python -m http.server 8080
```

Open http://localhost:8080. Camera input and geolocation work on `localhost` and
over HTTPS.

## Layout

| path | what |
|---|---|
| `index.html`, `style.css`, `app.js` | the app |
| `model/seed{42,43,44}.fp16.onnx` | 3-seed ResNet-18 ensemble, float16, about 22.6 MB each. seed42 loads first, 43 and 44 follow in the background |
| `model/labels.json` | 11 genus codes and names |
| `cloud-info.json` | per-genus description, altitude, weather meaning |
| `manifest.webmanifest`, `sw.js`, `icons/` | PWA install and offline cache |
| `tools/export_onnx.py` | rebuilds the ONNX models from the `.pth` checkpoints (`--ckpt-dir` or `$CCSN11_CKPT_DIR`) |

## Data sources

Weather comes from [Open-Meteo](https://open-meteo.com/), free, no key, using the
DWD ICON model (`icon_seamless`, highest resolution grid available for the spot).
Place names come from the BigDataCloud reverse-geocode client endpoint, free, no
key. Neither is cached by the service worker.

## Deploy

Static hosting works: GitHub Pages, Cloudflare Pages, or Netlify. Publish the
`app/` folder only. See [`DEPLOY.md`](DEPLOY.md). The `app/` folder is its own git
repo with the remote `cloud-genus`, so run its git commands from inside `app/`,
not from the research repo root. onnxruntime-web loads from the jsdelivr CDN and
is cached for offline use after the first visit.

## Known limits

The model is trained on CCSN ground close-ups, so phone framing scores lower, and
it is weak on flat overcast (St, As, Cs). The not-a-cloud check is a simple
brightness and edge test at about 512 px. Most real sky photos pass, and you can
override it with "classify anyway". The advisory is guidance from the cloud type
and current conditions, not an official forecast or severe-weather warning.
