export const content = `
# Olay İşleme Sistemi Alıştırması

Bu alıştırmada, farklı olay türlerini işleyebilen bir sistem tasarlayacağız. Sistem, soyut sınıflar ve arayüzler kullanarak olay işleme işlevselliğini sağlayacak.

## Problem Tanımı

Aşağıdaki özelliklere sahip bir olay işleme sistemi geliştirmeniz gerekiyor:

1. Farklı olay türleri için işleyiciler (sistem olayları, kullanıcı olayları, hata olayları vb.)
2. Olay önceliklendirme
3. Olay filtreleme
4. Olay günlüğü tutma
5. Asenkron olay işleme

## Çözüm

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from queue import PriorityQueue
import asyncio
import logging

# Olay türleri için enum
class EventType(Enum):
    SYSTEM = auto()
    USER = auto()
    ERROR = auto()
    NETWORK = auto()
    CUSTOM = auto()

# Olay önceliği için enum
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Temel olay sınıfı
@dataclass
class Event:
    type: EventType
    name: str
    data: Any
    timestamp: datetime = datetime.now()
    priority: Priority = Priority.MEDIUM
    source: Optional[str] = None

# Olay işleyici protokolü
class EventHandler(ABC):
    @abstractmethod
    def handle_event(self, event: Event) -> bool:
        pass

# Sistem olayları işleyicisi
class SystemEventHandler(EventHandler):
    def handle_event(self, event: Event) -> bool:
        if event.type != EventType.SYSTEM:
            return False
        
        print(f"Sistem olayı işleniyor: {event.name}")
        print(f"Veri: {event.data}")
        print(f"Zaman: {event.timestamp}")
        return True

# Kullanıcı olayları işleyicisi
class UserEventHandler(EventHandler):
    def handle_event(self, event: Event) -> bool:
        if event.type != EventType.USER:
            return False
        
        print(f"Kullanıcı olayı işleniyor: {event.name}")
        print(f"Kullanıcı: {event.source}")
        print(f"Veri: {event.data}")
        return True

# Hata olayları işleyicisi
class ErrorEventHandler(EventHandler):
    def handle_event(self, event: Event) -> bool:
        if event.type != EventType.ERROR:
            return False
        
        error_data = event.data
        print(f"Hata olayı işleniyor: {event.name}")
        print(f"Hata mesajı: {error_data.get('message')}")
        print(f"Hata kodu: {error_data.get('code')}")
        
        # Hata günlüğüne kaydet
        logging.error(f"Hata: {event.name} - {error_data}")
        return True

# Ağ olayları işleyicisi
class NetworkEventHandler(EventHandler):
    def handle_event(self, event: Event) -> bool:
        if event.type != EventType.NETWORK:
            return False
        
        print(f"Ağ olayı işleniyor: {event.name}")
        print(f"Durum: {event.data.get('status')}")
        print(f"Adres: {event.data.get('address')}")
        return True

# Olay filtresi
class EventFilter:
    def __init__(self):
        self.type_filters: Set[EventType] = set()
        self.priority_filters: Set[Priority] = set()
        self.name_filters: Set[str] = set()
        self.source_filters: Set[str] = set()
    
    def add_type_filter(self, event_type: EventType):
        self.type_filters.add(event_type)
    
    def add_priority_filter(self, priority: Priority):
        self.priority_filters.add(priority)
    
    def add_name_filter(self, name: str):
        self.name_filters.add(name)
    
    def add_source_filter(self, source: str):
        self.source_filters.add(source)
    
    def matches(self, event: Event) -> bool:
        if self.type_filters and event.type not in self.type_filters:
            return False
        
        if self.priority_filters and event.priority not in self.priority_filters:
            return False
        
        if self.name_filters and event.name not in self.name_filters:
            return False
        
        if self.source_filters and event.source not in self.source_filters:
            return False
        
        return True

# Olay günlüğü
class EventLogger:
    def __init__(self, filename: str = "events.log"):
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_event(self, event: Event):
        log_message = (
            f"Olay: {event.name}, "
            f"Tür: {event.type.name}, "
            f"Öncelik: {event.priority.name}, "
            f"Kaynak: {event.source}"
        )
        logging.info(log_message)

# Olay yöneticisi
class EventManager:
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.filters: List[EventFilter] = []
        self.logger = EventLogger()
        self.event_queue = PriorityQueue()
        
        # Her olay türü için boş bir işleyici listesi oluştur
        for event_type in EventType:
            self.handlers[event_type] = []
    
    def register_handler(self, handler: EventHandler, event_type: EventType):
        self.handlers[event_type].append(handler)
    
    def add_filter(self, event_filter: EventFilter):
        self.filters.append(event_filter)
    
    def publish_event(self, event: Event):
        # Olayı kuyruğa ekle (öncelik değeri ile)
        self.event_queue.put((-event.priority.value, event))
    
    def _should_process_event(self, event: Event) -> bool:
        # Hiç filtre yoksa veya en az bir filtre eşleşiyorsa işle
        return not self.filters or any(f.matches(event) for f in self.filters)
    
    async def process_events(self):
        while True:
            if not self.event_queue.empty():
                # Kuyruktaki en yüksek öncelikli olayı al
                _, event = self.event_queue.get()
                
                # Filtreleri kontrol et
                if not self._should_process_event(event):
                    continue
                
                # Olayı günlüğe kaydet
                self.logger.log_event(event)
                
                # İlgili işleyicileri çağır
                for handler in self.handlers[event.type]:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, handler.handle_event, event
                        )
                    except Exception as e:
                        logging.error(f"Olay işleme hatası: {e}")
            
            await asyncio.sleep(0.1)  # CPU kullanımını azaltmak için kısa bekle

# Kullanım örneği
async def main():
    # Olay yöneticisi oluştur
    manager = EventManager()
    
    # İşleyicileri kaydet
    manager.register_handler(SystemEventHandler(), EventType.SYSTEM)
    manager.register_handler(UserEventHandler(), EventType.USER)
    manager.register_handler(ErrorEventHandler(), EventType.ERROR)
    manager.register_handler(NetworkEventHandler(), EventType.NETWORK)
    
    # Filtre oluştur ve ekle
    high_priority_filter = EventFilter()
    high_priority_filter.add_priority_filter(Priority.HIGH)
    high_priority_filter.add_priority_filter(Priority.CRITICAL)
    manager.add_filter(high_priority_filter)
    
    # Örnek olaylar oluştur ve yayınla
    events = [
        Event(
            type=EventType.SYSTEM,
            name="SystemStartup",
            data={"status": "initializing"},
            priority=Priority.HIGH
        ),
        Event(
            type=EventType.USER,
            name="UserLogin",
            data={"user_id": "12345"},
            source="web_app",
            priority=Priority.MEDIUM
        ),
        Event(
            type=EventType.ERROR,
            name="DatabaseError",
            data={"message": "Connection failed", "code": 500},
            priority=Priority.CRITICAL
        ),
        Event(
            type=EventType.NETWORK,
            name="NetworkTimeout",
            data={"status": "timeout", "address": "api.example.com"},
            priority=Priority.HIGH
        )
    ]
    
    # Olayları yayınla
    for event in events:
        manager.publish_event(event)
    
    # Olay işleme döngüsünü başlat
    await manager.process_events()

# Programı çalıştır
if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Önemli Noktlar

1. **Olay Türleri**: Farklı olay türleri için ayrı işleyiciler tanımlanmıştır.
2. **Önceliklendirme**: Olaylar öncelik sırasına göre işlenir.
3. **Filtreleme**: Olaylar türe, önceliğe, ada ve kaynağa göre filtrelenebilir.
4. **Günlük Tutma**: Tüm olaylar için detaylı günlük kaydı tutulur.
5. **Asenkron İşleme**: Olaylar asenkron olarak işlenir.

## Geliştirme Önerileri

1. **Dağıtık Sistem Desteği**: Farklı sunucular arasında olay senkronizasyonu.
2. **Olay Geçmişi**: Geçmiş olayları sorgulama ve analiz etme özellikleri.
3. **İzleme Paneli**: Gerçek zamanlı olay izleme için web arayüzü.
4. **Olay Şemaları**: Farklı olay türleri için şema doğrulama.
5. **Otomatik Ölçeklendirme**: Yüksek olay yükü için otomatik ölçeklendirme.

Bu örnek, soyut sınıflar ve arayüzlerin olay işleme gibi karmaşık sistemlerde nasıl kullanılabileceğini göstermektedir. Sistem, yeni olay türleri ve işleyiciler eklemek için kolayca genişletilebilir şekilde tasarlanmıştır.
`; 