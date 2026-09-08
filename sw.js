/* Cloud Genus service worker — offline app shell + model, runtime-cache the rest */
const CACHE = 'cloud-genus-v5';
const SHELL = [
  './',
  'index.html',
  'style.css',
  'app.js',
  'cloud-info.json',
  'manifest.webmanifest',
  'model/labels.json',
  'model/seed42.fp16.onnx',
  'model/seed43.fp16.onnx',
  'model/seed44.fp16.onnx',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const { request } = e;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);

  // never cache the weather / geocode APIs
  if (url.hostname.includes('open-meteo.com') || url.hostname.includes('bigdatacloud.net')) return;

  // cache-first for everything else (app shell, model, onnxruntime CDN + wasm)
  e.respondWith(
    caches.match(request).then(
      (hit) =>
        hit ||
        fetch(request).then((res) => {
          if (res.ok && (url.origin === location.origin || url.hostname.includes('jsdelivr.net'))) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(request, copy));
          }
          return res;
        }).catch(() => hit)
    )
  );
});
