import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Progressive Web Apps | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamalarını PWA\'ya dönüştürme. Service workers, offline functionality, push notifications ve modern web özellikleri.',
};

const content = `
# Progressive Web Apps

Modern web uygulamalarını PWA'ya dönüştürmeyi ve Python backend ile entegrasyonunu öğreneceğiz.

## Service Worker Kurulumu

\`\`\`javascript
// public/service-worker.js
const CACHE_NAME = 'app-cache-v1';
const URLS_TO_CACHE = [
  '/',
  '/offline',
  '/static/css/main.css',
  '/static/js/main.js',
  '/static/images/logo.png'
];

// Service Worker kurulumu
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Cache opened');
        return cache.addAll(URLS_TO_CACHE);
      })
  );
});

// Fetch olaylarını yakalama
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        return fetch(event.request).then(
          (response) => {
            // Geçersiz yanıt kontrolü
            if(!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            // Cache'e kopyalama
            const responseToCache = response.clone();
            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(event.request, responseToCache);
              });

            return response;
          }
        ).catch(() => {
          // Offline sayfasına yönlendirme
          return caches.match('/offline');
        });
      })
  );
});

// Push notification olaylarını yakalama
self.addEventListener('push', (event) => {
  const options = {
    body: event.data.text(),
    icon: '/static/images/icon.png',
    badge: '/static/images/badge.png'
  };

  event.waitUntil(
    self.registration.showNotification('Yeni Bildirim', options)
  );
});
\`\`\`

## PWA Manifest

\`\`\`json
// public/manifest.json
{
  "name": "Python Web App",
  "short_name": "PyApp",
  "description": "Modern Python Web Application",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#000000",
  "icons": [
    {
      "src": "/static/images/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/images/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
\`\`\`

## Backend Push Notification Servisi

\`\`\`python
# services/push_notification.py
from typing import Dict, List
import json
from pywebpush import webpush, WebPushException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class PushSubscription(BaseModel):
    endpoint: str
    keys: Dict[str, str]

class NotificationPayload(BaseModel):
    title: str
    body: str
    icon: str = None
    data: Dict = None

class PushService:
    def __init__(self, vapid_private_key: str, vapid_claims: Dict):
        self.vapid_private_key = vapid_private_key
        self.vapid_claims = vapid_claims
        self.subscriptions: List[PushSubscription] = []
    
    async def add_subscription(self, subscription: PushSubscription):
        self.subscriptions.append(subscription)
    
    async def remove_subscription(self, endpoint: str):
        self.subscriptions = [
            s for s in self.subscriptions
            if s.endpoint != endpoint
        ]
    
    async def send_notification(
        self,
        payload: NotificationPayload,
        subscription: PushSubscription
    ):
        try:
            webpush(
                subscription_info={
                    "endpoint": subscription.endpoint,
                    "keys": subscription.keys
                },
                data=json.dumps(payload.dict()),
                vapid_private_key=self.vapid_private_key,
                vapid_claims=self.vapid_claims
            )
        except WebPushException as e:
            if e.response.status_code == 410:
                # Subscription expired
                await self.remove_subscription(subscription.endpoint)
            raise HTTPException(
                status_code=500,
                detail=f"Push notification error: {str(e)}"
            )

push_service = PushService(
    vapid_private_key="your-vapid-private-key",
    vapid_claims={
        "sub": "mailto:your-email@example.com"
    }
)

@app.post("/push/subscribe")
async def subscribe(subscription: PushSubscription):
    await push_service.add_subscription(subscription)
    return {"message": "Subscription added"}

@app.post("/push/notify")
async def notify(payload: NotificationPayload):
    for subscription in push_service.subscriptions:
        await push_service.send_notification(payload, subscription)
    return {"message": "Notifications sent"}
\`\`\`

## Offline Storage ve Senkronizasyon

\`\`\`typescript
// lib/storage.ts
import { openDB, DBSchema, IDBPDatabase } from 'idb';

interface TodoDB extends DBSchema {
  todos: {
    key: string;
    value: {
      id: string;
      title: string;
      completed: boolean;
      synced: boolean;
    };
    indexes: { 'by-synced': boolean };
  };
}

class OfflineStorage {
  private db: IDBPDatabase<TodoDB>;

  async init() {
    this.db = await openDB<TodoDB>('todo-db', 1, {
      upgrade(db) {
        const store = db.createObjectStore('todos', {
          keyPath: 'id'
        });
        store.createIndex('by-synced', 'synced');
      }
    });
  }

  async addTodo(todo: Omit<TodoDB['todos']['value'], 'synced'>) {
    await this.db.add('todos', { ...todo, synced: false });
    this.syncWithServer();
  }

  async getTodos() {
    return this.db.getAll('todos');
  }

  async syncWithServer() {
    const unsynced = await this.db.getAllFromIndex(
      'todos',
      'by-synced',
      false
    );

    for (const todo of unsynced) {
      try {
        await fetch('/api/todos', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(todo)
        });

        await this.db.put('todos', { ...todo, synced: true });
      } catch (error) {
        console.error('Sync error:', error);
      }
    }
  }
}

export const storage = new OfflineStorage();
\`\`\`

## Alıştırmalar

1. **Service Worker**
   - Cache stratejileri implementasyonu yapın
   - Offline sayfası tasarlayın
   - Background sync ekleyin
   - Push notification entegrasyonu yapın

2. **PWA Özellikleri**
   - App manifest oluşturun
   - Splash screen tasarlayın
   - Home screen installation ekleyin
   - Offline storage implementasyonu yapın

3. **Backend Entegrasyonu**
   - Push notification servisi kurun
   - Offline senkronizasyon ekleyin
   - WebSocket bağlantısı kurun
   - API caching stratejisi belirleyin

## Sonraki Adımlar

1. [DevOps Practices](/topics/python/web-gelistirme/devops)
2. [Security Best Practices](/topics/python/web-gelistirme/security)
3. [Performance Optimization](/topics/python/web-gelistirme/performance)

## Faydalı Kaynaklar

- [PWA Documentation](https://web.dev/progressive-web-apps/)
- [Service Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
- [Web Push Protocol](https://developers.google.com/web/fundamentals/push-notifications)
- [IndexedDB API](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)
`;

const learningPath = [
  {
    title: '1. Service Workers',
    description: 'Service worker ve caching stratejilerini öğrenin.',
    topics: [
      'Service worker yaşam döngüsü',
      'Cache stratejileri',
      'Offline functionality',
      'Background sync',
      'Push notifications',
    ],
    icon: '🔄',
    href: '/topics/python/web-gelistirme/pwa/service-workers'
  },
  {
    title: '2. PWA Features',
    description: 'PWA özelliklerini ve implementasyonunu öğrenin.',
    topics: [
      'App manifest',
      'Install prompts',
      'Splash screens',
      'Home screen icons',
      'App shell architecture',
    ],
    icon: '📱',
    href: '/topics/python/web-gelistirme/pwa/features'
  },
  {
    title: '3. Offline Storage',
    description: 'Offline storage ve senkronizasyonu öğrenin.',
    topics: [
      'IndexedDB',
      'Cache API',
      'Background sync',
      'Conflict resolution',
      'Data persistence',
    ],
    icon: '💾',
    href: '/topics/python/web-gelistirme/pwa/storage'
  },
  {
    title: '4. Push Notifications',
    description: 'Push notification sistemini öğrenin.',
    topics: [
      'Web Push API',
      'Notification API',
      'Service worker events',
      'Push server',
      'User permissions',
    ],
    icon: '🔔',
    href: '/topics/python/web-gelistirme/pwa/notifications'
  }
];

export default function PWAPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/web-gelistirme" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Web Geliştirme
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <h2 className="text-2xl font-bold mb-6">Öğrenme Yolu</h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          {learningPath.map((topic, index) => (
            <Card key={index} className="p-6 hover:bg-accent transition-colors cursor-pointer">
              <Link href={topic.href}>
                <div className="flex items-start space-x-4">
                  <div className="text-4xl">{topic.icon}</div>
                  <div className="space-y-2">
                    <h3 className="font-bold">{topic.title}</h3>
                    <p className="text-sm text-muted-foreground">{topic.description}</p>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {topic.topics.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Link>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 