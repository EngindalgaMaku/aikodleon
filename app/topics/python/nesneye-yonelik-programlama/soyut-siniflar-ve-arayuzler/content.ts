export const content = `
# Python'da Soyut Sınıflar ve Arayüzler

Python'da soyut sınıflar ve arayüzler, nesne yönelimli programlamada kod organizasyonu ve tasarım için önemli araçlardır. Bu yapılar, sınıflar arasında sözleşmeler tanımlamamızı ve kodun daha düzenli olmasını sağlar.

## Soyut Sınıflar (Abstract Base Classes)

Soyut sınıflar, doğrudan örneklenemeyen ve alt sınıflar için bir şablon görevi gören sınıflardır. Python'da soyut sınıflar \`abc\` modülü ile oluşturulur.

\`\`\`python
from abc import ABC, abstractmethod

class VeriTabani(ABC):
    @abstractmethod
    def baglan(self) -> bool:
        """Veritabanına bağlanır."""
        pass
    
    @abstractmethod
    def sorgu_calistir(self, sorgu: str) -> list:
        """Verilen sorguyu çalıştırır ve sonuçları döndürür."""
        pass
    
    @abstractmethod
    def baglantiyi_kapat(self) -> bool:
        """Veritabanı bağlantısını kapatır."""
        pass
    
    def test_baglantisi(self) -> bool:
        """Veritabanı bağlantısını test eder."""
        try:
            return self.baglan() and self.baglantiyi_kapat()
        except Exception:
            return False

class PostgreSQL(VeriTabani):
    def __init__(self, host: str, port: int, kullanici: str, sifre: str):
        self.host = host
        self.port = port
        self.kullanici = kullanici
        self.sifre = sifre
        self.baglanti = None
    
    def baglan(self) -> bool:
        print(f"PostgreSQL'e bağlanılıyor: {self.host}:{self.port}")
        self.baglanti = True  # Gerçek uygulamada psycopg2 gibi bir kütüphane kullanılır
        return True
    
    def sorgu_calistir(self, sorgu: str) -> list:
        if not self.baglanti:
            raise Exception("Veritabanı bağlantısı yok!")
        print(f"Sorgu çalıştırılıyor: {sorgu}")
        return [{"id": 1, "ad": "Test"}]  # Örnek veri
    
    def baglantiyi_kapat(self) -> bool:
        if self.baglanti:
            print("PostgreSQL bağlantısı kapatılıyor")
            self.baglanti = None
            return True
        return False

class MongoDB(VeriTabani):
    def __init__(self, uri: str):
        self.uri = uri
        self.baglanti = None
    
    def baglan(self) -> bool:
        print(f"MongoDB'ye bağlanılıyor: {self.uri}")
        self.baglanti = True  # Gerçek uygulamada pymongo kullanılır
        return True
    
    def sorgu_calistir(self, sorgu: str) -> list:
        if not self.baglanti:
            raise Exception("Veritabanı bağlantısı yok!")
        print(f"Sorgu çalıştırılıyor: {sorgu}")
        return [{"_id": 1, "name": "Test"}]  # Örnek veri
    
    def baglantiyi_kapat(self) -> bool:
        if self.baglanti:
            print("MongoDB bağlantısı kapatılıyor")
            self.baglanti = None
            return True
        return False

# Kullanım
def veritabani_islemleri(db: VeriTabani):
    if db.baglan():
        sonuclar = db.sorgu_calistir("SELECT * FROM users")
        print(f"Sorgu sonuçları: {sonuclar}")
        db.baglantiyi_kapat()

# Her iki veritabanı ile de çalışır
postgres_db = PostgreSQL("localhost", 5432, "admin", "123456")
mongo_db = MongoDB("mongodb://localhost:27017")

veritabani_islemleri(postgres_db)
veritabani_islemleri(mongo_db)
\`\`\`

**Soyut Sınıfların Avantajları**

- **Zorunlu Metodlar:** Alt sınıfların belirli metodları uygulamasını zorunlu kılar.
- **Tip Güvenliği:** Alt sınıfların arayüzünü garanti eder.
- **Kod Organizasyonu:** Ortak davranışları tek bir yerde toplar.
- **Tasarım Rehberi:** Alt sınıflar için bir şablon sağlar.

## Protokoller (Protocols)

Python 3.8 ile birlikte gelen protokoller, yapısal alt tipleme sağlayan bir özelliktir. Protokoller, bir sınıfın belirli metodlara sahip olmasını bekler, ancak doğrudan kalıtım gerektirmez.

\`\`\`python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Yazilabilir(Protocol):
    def yaz(self, veri: str) -> bool:
        ...

class DosyaYazici:
    def yaz(self, veri: str) -> bool:
        print(f"Dosyaya yazılıyor: {veri}")
        return True

class BellekYazici:
    def yaz(self, veri: str) -> bool:
        print(f"Belleğe yazılıyor: {veri}")
        return True

def veri_kaydet(hedef: Yazilabilir, veri: str):
    if hedef.yaz(veri):
        print("Veri başarıyla kaydedildi")
    else:
        print("Veri kaydedilemedi")

# Her iki sınıf da Yazilabilir protokolünü uygular
dosya = DosyaYazici()
bellek = BellekYazici()

veri_kaydet(dosya, "Test verisi")
veri_kaydet(bellek, "Test verisi")

# Protokol kontrolü
print(isinstance(dosya, Yazilabilir))  # True
print(isinstance(bellek, Yazilabilir))  # True
\`\`\`

**Protokoller vs Soyut Sınıflar**

- **Protokoller:**
  - Yapısal alt tipleme (duck typing)
  - Kalıtım gerektirmez
  - Çalışma zamanı kontrolü (\`@runtime_checkable\` ile)
  - Hafif ve esnek
- **Soyut Sınıflar:**
  - Nominal alt tipleme
  - Kalıtım gerektirir
  - Derleme zamanı kontrolü
  - Daha katı ve resmi

## Pratik Örnek: Oyun Motoru

Aşağıdaki örnek, bir oyun motorunda soyut sınıflar ve protokollerin nasıl kullanılabileceğini gösterir:

\`\`\`python
from abc import ABC, abstractmethod
from typing import Protocol, List, Tuple
import math

# Temel protokoller
class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Collidable(Protocol):
    def check_collision(self, other: 'Collidable') -> bool:
        ...

# Temel soyut sınıf
class GameObject(ABC):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    @abstractmethod
    def update(self, delta_time: float) -> None:
        pass
    
    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

# Sprite sınıfı
class Sprite(GameObject, Drawable, Collidable):
    def __init__(self, x: float, y: float, width: float, height: float):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.velocity_x = 0
        self.velocity_y = 0
    
    def update(self, delta_time: float) -> None:
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time
    
    def draw(self) -> None:
        print(f"Drawing sprite at ({self.x}, {self.y})")
    
    def check_collision(self, other: Collidable) -> bool:
        if isinstance(other, Sprite):
            return (self.x < other.x + other.width and
                    self.x + self.width > other.x and
                    self.y < other.y + other.height and
                    self.y + self.height > other.y)
        return False

# Oyun dünyası
class GameWorld:
    def __init__(self):
        self.objects: List[GameObject] = []
        self.drawables: List[Drawable] = []
        self.collidables: List[Collidable] = []
    
    def add_object(self, obj: GameObject) -> None:
        self.objects.append(obj)
        if isinstance(obj, Drawable):
            self.drawables.append(obj)
        if isinstance(obj, Collidable):
            self.collidables.append(obj)
    
    def update(self, delta_time: float) -> None:
        # Tüm nesneleri güncelle
        for obj in self.objects:
            obj.update(delta_time)
        
        # Çarpışmaları kontrol et
        for i, obj1 in enumerate(self.collidables):
            for obj2 in self.collidables[i+1:]:
                if obj1.check_collision(obj2):
                    print(f"Collision detected between {obj1} and {obj2}")
    
    def draw(self) -> None:
        for drawable in self.drawables:
            drawable.draw()

# Kullanım örneği
def main():
    # Oyun dünyası oluştur
    world = GameWorld()
    
    # Nesneler ekle
    player = Sprite(100, 100, 50, 50)
    player.velocity_x = 10
    
    enemy = Sprite(200, 100, 50, 50)
    enemy.velocity_x = -5
    
    world.add_object(player)
    world.add_object(enemy)
    
    # Oyun döngüsü
    for _ in range(3):  # 3 kare simüle et
        world.update(0.016)  # ~60 FPS
        world.draw()
        print("---")

if __name__ == "__main__":
    main()
\`\`\`

## Alıştırmalar

1. [Medya İşleme Sistemi](/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler/medya-isleme)
   - Farklı medya türleri (müzik, video, podcast) için ortak bir arayüz
   - Her medya türü için özel oynatma davranışları
   - Format dönüştürme ve kalite ayarları

2. [Veri Doğrulama Sistemi](/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler/veri-dogrulama)
   - Farklı veri türleri için doğrulama kuralları
   - Özelleştirilebilir hata mesajları
   - Zincirleme doğrulama kuralları

3. [Olay İşleme Sistemi](/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler/olay-isleme)
   - Farklı olay türleri için işleyiciler
   - Olay önceliklendirme ve filtreleme
   - Asenkron olay işleme

## Sonraki Adımlar

Soyut sınıflar ve arayüzler konusunu öğrendiniz. Bu yapıları kullanarak kodunuzu daha düzenli ve genişletilebilir hale getirebilirsiniz. Bir sonraki adım olarak tasarım desenlerini öğrenerek, bu yapıları en etkili şekilde nasıl kullanacağınızı görebilirsiniz.
`; 