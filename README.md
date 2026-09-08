# Cloud Genus — phone app (PWA)

Installable web app. Take a photo of the sky → cloud genus + what it means → your local
weather and a plain-language advisory. The classifier runs **on the device** (ONNX in the
browser); only the optional weather lookup needs a connection.

## Run locally

```
cd app
python -m http.server 8080
```

Open http://localhost:8080. Camera input and geolocation work on `localhost` and over HTTPS.

## Layout

| path | what |
|---|---|
| `index.html`, `style.css`, `app.js` | the app |
| `model/seed{42,43,44}.int8.onnx` | 3-seed ResNet-18 ensemble, int8-quantized (~11 MB each) |
| `model/labels.json` | 11 genus codes + names |
| `cloud-info.json` | per-genus description, altitude, weather meaning |
| `manifest.webmanifest`, `sw.js`, `icons/` | PWA install + offline cache |
| `tools/export_onnx.py` | regenerates the ONNX models from the `.pth` checkpoints |

## Data sources

- **Weather**: [Open-Meteo](https://open-meteo.com/) — free, no key.
- **Place name**: BigDataCloud reverse-geocode client endpoint — free, no key.
- Neither is cached by the service worker.

## Deploy

Static hosting — GitHub Pages, Cloudflare Pages, Netlify. Just publish the `app/` folder.
onnxruntime-web loads from the jsdelivr CDN (allowed) and is cached for offline use after
the first visit.

## Known limits (v1)

- Model is CCSN-trained (ground close-ups); phone framing will score lower. Weak on flat
  overcast (St, As, Cs). No "not a cloud" guard yet — a random photo still gets a guess.
- Advisory is guidance from the cloud type + current conditions, **not** an official
  forecast or severe-weather warning.
