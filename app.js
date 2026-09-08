/* Cloud Genus — on-device cloud classifier + local weather context */
'use strict';

const CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
ort.env.wasm.wasmPaths = CDN;
ort.env.wasm.numThreads = 1;

const SEEDS = [42, 43, 44];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SIZE = 224;

const $ = (id) => document.getElementById(id);
let sessions = [];
let LABELS = [];
let INFO = {};
let lastBlob = null;
let lastTop = null;
let lastWx = null;
let modelReady = false;
let pendingFile = null;
let forceClassify = false;
let lastImg = null;

/* ---------- model ---------- */


async function loadModel() {
  const [labels, info] = await Promise.all([
    fetch('model/labels.json').then((r) => r.json()),
    fetch('cloud-info.json').then((r) => r.json()),
  ]);
  LABELS = labels;
  INFO = info;
  sessions = await Promise.all(
    SEEDS.map((s) =>
      ort.InferenceSession.create(`model/seed${s}.fp16.onnx`, {
        executionProviders: ['wasm'],
      })
    )
  );
  modelReady = true;
  $('cam').disabled = $('gal').disabled = false;
  document.querySelectorAll('#capture label').forEach((l) => l.classList.remove('disabled'));
  if (pendingFile) {
    const f = pendingFile;
    pendingFile = null;
    handleFile(f);
  }
}

function preprocess(img) {
  const c = document.createElement('canvas');
  c.width = SIZE;
  c.height = SIZE;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const hw = SIZE * SIZE;
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    out[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    out[hw + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    out[2 * hw + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', out, [1, 3, SIZE, SIZE]);
}

/* rough "is this the sky?" check — soft, over-rideable */
function skyCheck(img) {
  const n = 160;
  const c = document.createElement('canvas');
  c.width = c.height = n;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, n, n);
  const d = ctx.getImageData(0, 0, n, n).data;

  let sat = 0;
  let edge = 0;
  const bright = new Float64Array(n * n);
  let rgSum = 0, ybSum = 0, rgSq = 0, ybSq = 0;
  for (let i = 0, p = 0; i < n * n; i++, p += 4) {
    const r = d[p], g = d[p + 1], b = d[p + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    sat += mx > 0 ? (mx - mn) / mx : 0;
    bright[i] = mx;
    const rg = Math.abs(r - g);
    const yb = Math.abs(0.5 * (r + g) - b);
    rgSum += rg; ybSum += yb; rgSq += rg * rg; ybSq += yb * yb;
  }
  const cnt = n * n;
  for (let y = 0; y < n; y++) {
    for (let x = 1; x < n; x++) edge += Math.abs(bright[y * n + x] - bright[y * n + x - 1]);
  }
  for (let y = 1; y < n; y++) {
    for (let x = 0; x < n; x++) edge += Math.abs(bright[y * n + x] - bright[(y - 1) * n + x]);
  }
  const meanSat = sat / cnt;
  const edgeDensity = edge / (2 * cnt);
  const rgStd = Math.sqrt(rgSq / cnt - (rgSum / cnt) ** 2);
  const ybStd = Math.sqrt(ybSq / cnt - (ybSum / cnt) ** 2);
  const colorfulness = Math.sqrt(rgStd ** 2 + ybStd ** 2) +
    0.3 * Math.sqrt((rgSum / cnt) ** 2 + (ybSum / cnt) ** 2);

  // clouds: edgeDensity < ~12, and never both very colorful and very saturated
  const ok = edgeDensity < 16 && !(colorfulness > 95 && meanSat > 0.6);
  return { ok };
}

function softmax(a) {
  const m = Math.max(...a);
  const e = a.map((x) => Math.exp(x - m));
  const s = e.reduce((p, x) => p + x, 0);
  return e.map((x) => x / s);
}

async function classify(img) {
  if (!sessions.length) throw new Error('model not loaded');
  const x = preprocess(img);
  const probs = new Array(LABELS.length).fill(0);
  for (const sess of sessions) {
    const out = await sess.run({ input: x });
    const logits = Array.from(out[Object.keys(out)[0]].data);
    softmax(logits).forEach((p, i) => (probs[i] += p / sessions.length));
  }
  return probs
    .map((p, i) => ({ ...LABELS[i], prob: p }))
    .sort((a, b) => b.prob - a.prob);
}

/* ---------- UI flow ---------- */

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  if (!modelReady) {
    pendingFile = file;
    $('capture').hidden = true;
    $('result').hidden = false;
    $('photo').src = URL.createObjectURL(file);
    $('verdict').textContent = 'loading the model…';
    $('verdict').classList.add('soft');
    $('summary').textContent = 'first run downloads ~68 MB, then it works offline';
    return;
  }
  lastBlob = file;
  forceClassify = false;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => run(img, url);
  img.onerror = () => {
    $('verdict').classList.remove('soft');
    $('verdict').textContent = "couldn't open that image";
  };
  img.src = url;
}

async function run(img, url) {
  lastImg = img;
  $('capture').hidden = true;
  $('result').hidden = false;
  $('photo').src = url;
  $('verdict').textContent = 'reading the sky…';
  $('verdict').classList.add('soft');
  $('summary').textContent = '';
  $('alert').hidden = true;
  $('correct').hidden = true;
  $('notsky').hidden = true;
  $('weather').hidden = true;

  if (!forceClassify && !skyCheck(img).ok) {
    $('verdict').textContent = 'Hmm…';
    $('notsky').hidden = false;
    return;
  }
  $('weather').hidden = false;

  let ranked;
  try {
    ranked = await classify(img);
  } catch (e) {
    $('verdict').textContent = 'could not read the model — check your connection and reload';
    return;
  }
  lastTop = ranked[0];
  const info = INFO[ranked[0].code];

  $('verdict').classList.remove('soft');
  $('verdict').textContent = info ? info.name : ranked[0].name;
  $('summary').textContent = info ? info.summary : '';

  const sel = $('truelabel');
  if (!sel.options.length) {
    [...LABELS]
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach((l) => sel.add(new Option(l.name, l.code)));
  }
  sel.value = ranked[0].code;
  $('savemsg').textContent = '';
  $('correct').hidden = false;

  // cloud-only alert straight away; refined once weather is in
  renderAlert(null);

  $('wx-body').hidden = true;
  $('wx-place').textContent = 'Local weather…';
  $('wx-enable').hidden = true;
  getWeather();
}

/* ---------- weather ---------- */

const WMO = {
  0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
  45: 'fog', 48: 'freezing fog', 51: 'light drizzle', 53: 'drizzle', 55: 'heavy drizzle',
  61: 'light rain', 63: 'rain', 65: 'heavy rain', 66: 'freezing rain', 67: 'freezing rain',
  71: 'light snow', 73: 'snow', 75: 'heavy snow', 77: 'snow grains',
  80: 'rain showers', 81: 'rain showers', 82: 'violent rain showers',
  85: 'snow showers', 86: 'heavy snow showers',
  95: 'thunderstorm', 96: 'thunderstorm with hail', 99: 'severe thunderstorm with hail',
};

function getWeather() {
  if (!navigator.geolocation) {
    $('wx-place').textContent = 'Location not available on this device';
    return;
  }
  $('wx-place').textContent = 'Getting your location…';
  $('wx-enable').hidden = true;
  navigator.geolocation.getCurrentPosition(
    (pos) => fetchWeather(pos.coords.latitude, pos.coords.longitude),
    (err) => {
      $('wx-place').textContent =
        err.code === 1 ? 'Location permission is off' : 'Could not get your location';
      $('wx-enable').hidden = false;
      $('wx-enable').textContent = 'retry';
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 120000 }
  );
}

async function fetchWeather(lat, lon, placeName) {
  try {
    const wx = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
        `&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,` +
        `weather_code,wind_speed_10m,wind_gusts_10m,cloud_cover&timezone=auto`
    ).then((r) => r.json());
    let place = placeName;
    if (!place) {
      place = await fetch(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`
      )
        .then((r) => r.json())
        .then((p) => [p.city || p.locality, p.principalSubdivision, p.countryName].filter(Boolean).join(', '))
        .catch(() => 'Your location');
    }
    renderWeather(wx.current, place);
  } catch (e) {
    $('wx-place').textContent = 'Weather unavailable right now';
    $('wx-enable').hidden = false;
    $('wx-enable').textContent = 'retry';
  }
}

function timeAgo(iso) {
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso + 'Z').getTime()) / 60000));
  if (mins < 60) return `updated ${mins} min ago`;
  return `updated ${Math.round(mins / 60)} h ago`;
}

function renderWeather(cur, place) {
  lastWx = cur;
  $('wx-enable').hidden = true;
  $('wx-body').hidden = false;
  $('wx-place').textContent = place || 'Your location';
  $('wx-updated').textContent = cur.time ? timeAgo(cur.time) : '';
  $('wx-temp').textContent = `${Math.round(cur.temperature_2m)}°`;
  $('wx-desc').textContent = WMO[cur.weather_code] || 'current conditions';

  const rows = [
    ['Feels like', `${Math.round(cur.apparent_temperature)}°`],
    ['Humidity', `${Math.round(cur.relative_humidity_2m)}%`],
    ['Wind', `${Math.round(cur.wind_speed_10m)} km/h`],
    ['Gusts', `${Math.round(cur.wind_gusts_10m)} km/h`],
    ['Cloud cover', `${Math.round(cur.cloud_cover)}%`],
    ['Precipitation', `${cur.precipitation ?? 0} mm`],
  ];
  $('wx-detail').innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join('');

  $('wx-note').textContent = softNote(lastTop.code, cur);
  renderAlert(cur);
}

/* severe-weather / cloud alert banner */
function renderAlert(cur) {
  const code = lastTop.code;
  const wc = cur ? cur.weather_code : null;
  const gust = cur ? cur.wind_gusts_10m : 0;
  const precip = cur ? cur.precipitation : 0;

  let level = 'none';
  let title = '';
  let text = '';

  const stormNow = [95, 96, 99].includes(wc);
  const heavyNow = [65, 75, 82, 86].includes(wc) || precip > 4;

  if (stormNow || (code === 'Cb' && (precip > 0 || gust > 45))) {
    level = 'severe';
    title = 'Take shelter';
    text =
      'Thunderstorm conditions. Get indoors, away from windows, open ground, water and tall isolated objects, and stay put until it clears.';
  } else if (code === 'Cb') {
    level = 'watch';
    title = 'Storm cloud overhead';
    text =
      'Cumulonimbus can bring lightning, hail and sudden gusts with little warning. Keep watching it and be ready to move indoors.';
  } else if (gust > 60) {
    level = 'severe';
    title = 'Strong winds';
    text = `Gusts around ${Math.round(gust)} km/h. Watch for falling branches and secure loose objects.`;
  } else if (code === 'Ns' && (precip > 0 || heavyNow)) {
    level = 'watch';
    title = 'Steady rain set in';
    text = 'Prolonged rain or snow is likely to continue for a while. Expect wet roads and low visibility.';
  } else if (heavyNow) {
    level = 'watch';
    title = 'Heavy precipitation';
    text = 'Heavy rain or snow right now. Take care on the roads.';
  }

  const box = $('alert');
  if (level === 'none') {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.className = 'alert' + (level === 'severe' ? ' severe' : '');
  box.innerHTML = `<b>${title}</b>${text}`;
}

function softNote(code, cur) {
  const raining = cur.precipitation > 0 || [51, 53, 55, 61, 63, 80, 81].includes(cur.weather_code);
  if (code === 'Cs' || code === 'As') {
    return raining
      ? 'The front this cloud belongs to has arrived.'
      : 'This often comes ahead of a warm front — rain or snow is plausible within the next 12–24 hours.';
  }
  if (code === 'Ci' || code === 'Cc') {
    return 'High, fair-weather cloud. Worth noting if it steadily thickens over the next few hours.';
  }
  if (code === 'Cu') {
    return cur.temperature_2m > 20 && cur.cloud_cover < 60
      ? 'Fair now. On a warm day, watch whether the puffs grow tall through the afternoon.'
      : 'Fair-weather cloud. No concern unless it builds upward.';
  }
  if (code === 'St' || code === 'Sc') {
    return 'A stable, mostly dry sky. It may stay grey a while, but a sudden change is unlikely.';
  }
  if (code === 'Ct') {
    return 'Not a weather cloud. Persistent, spreading contrails hint at moist air aloft.';
  }
  return raining ? 'Rain is falling from lower cloud than this.' : 'Settled conditions for now.';
}

/* ---------- save for fine-tuning (local) ---------- */

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('cloud-genus', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('samples', { keyPath: 'ts' });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function saveSample() {
  if (!lastBlob) return;
  try {
    const db = await idb();
    await new Promise((res, rej) => {
      const tx = db.transaction('samples', 'readwrite');
      tx.objectStore('samples').put({
        ts: Date.now(),
        label: $('truelabel').value,
        predicted: lastTop.code,
        blob: lastBlob,
      });
      tx.oncomplete = res;
      tx.onerror = () => rej(tx.error);
    });
    $('savemsg').textContent = 'saved on this device ✓';
  } catch (e) {
    $('savemsg').textContent = 'could not save';
  }
}

/* ---------- wire up ---------- */

$('cam').addEventListener('change', (e) => handleFile(e.target.files[0]));
$('gal').addEventListener('change', (e) => handleFile(e.target.files[0]));
$('again').addEventListener('click', () => {
  $('result').hidden = true;
  $('capture').hidden = false;
  $('cam').value = $('gal').value = '';
});
$('anyway').addEventListener('click', () => {
  forceClassify = true;
  if (lastImg) run(lastImg, $('photo').src);
});
$('wx-enable').addEventListener('click', getWeather);
$('savebtn').addEventListener('click', saveSample);

$('cam').disabled = $('gal').disabled = true;
loadModel().catch((e) => {
  console.error(e);
  const l = document.querySelector('.lede');
  if (l) l.textContent = 'Could not load the model. Reload with a connection.';
  if (pendingFile) {
    $('verdict').classList.remove('soft');
    $('verdict').textContent = 'model failed to load — reload the page';
  }
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
