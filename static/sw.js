const CACHE_NAME = 'dashboard-cache-v1';
const urlsToCache = [
  '/',
  '/static/manifest.json',
  '/static/icon.png',
  '/_dash-layout',
  '/_dash-dependencies'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('dash-pwa-v1').then((cache) => {
            return cache.addAll([
                '/',
                '/static/icon-144.png',
                '/static/icon-512.png',
                '/static/manifest.json'
            ]);
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        })
    );
});

self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (!cacheWhitelist.includes(cacheName)) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});