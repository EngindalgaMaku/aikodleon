import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Python Temel Veri Yapıları | Kodleon',
  description: 'Python\'da temel veri yapılarını, listeler, tuple\'lar, dictionary\'ler ve daha fazlasını öğrenin.',
};

const content = `
# Python'da Temel Veri Yapıları

Python'da temel veri yapıları, verileri organize etmek ve yönetmek için kullanılan yapı taşlarıdır. Bu bölümde, en sık kullanılan veri yapılarını ve bunların kullanım senaryolarını öğreneceğiz.

## 1. Listeler (Lists)

Listeler, Python'da en çok kullanılan veri yapılarından biridir. Farklı veri tiplerini içerebilen, sıralı ve değiştirilebilir koleksiyonlardır.

\`\`\`python
# Liste oluşturma
sayilar = [1, 2, 3, 4, 5]
karisik = [1, "Python", 3.14, True]

# Liste işlemleri
sayilar.append(6)        # Sona eleman ekleme
sayilar.insert(0, 0)    # Belirli konuma ekleme
sayilar.pop()           # Son elemanı çıkarma
sayilar.remove(3)       # Belirli elemanı çıkarma
sayilar.sort()          # Sıralama
sayilar.reverse()       # Ters çevirme

# Liste dilimleme
ilk_uc = sayilar[:3]    # İlk üç eleman
son_uc = sayilar[-3:]   # Son üç eleman
\`\`\`

## 2. Tuple'lar

Tuple'lar listelere benzer ancak değiştirilemez (immutable) yapılardır. Genellikle ilişkili verileri gruplamak için kullanılır.

\`\`\`python
# Tuple oluşturma
koordinat = (10, 20)
rgb = (255, 128, 0)

# Tuple işlemleri
x, y = koordinat        # Tuple unpacking
r, g, b = rgb

# Tuple metodları
indeks = rgb.index(128) # Eleman indeksini bulma
sayi = rgb.count(255)   # Eleman sayısını bulma
\`\`\`

## 3. Dictionary'ler

Dictionary'ler (sözlükler), anahtar-değer çiftlerini saklayan veri yapılarıdır. Hızlı erişim ve veri organizasyonu için idealdir.

\`\`\`python
# Dictionary oluşturma
ogrenci = {
    "ad": "Ahmet",
    "soyad": "Yılmaz",
    "yas": 20,
    "dersler": ["Python", "Matematik", "Fizik"]
}

# Dictionary işlemleri
ogrenci["bolum"] = "Bilgisayar"  # Yeni çift ekleme
del ogrenci["yas"]               # Çift silme
dersler = ogrenci.get("dersler") # Değer alma
anahtarlar = ogrenci.keys()      # Anahtarları alma
degerler = ogrenci.values()      # Değerleri alma
\`\`\`

## 4. Set'ler

Set'ler, benzersiz elemanları saklamak için kullanılan veri yapılarıdır. Matematikteki küme kavramına benzer şekilde çalışır.

\`\`\`python
# Set oluşturma
sayilar = {1, 2, 3, 4, 5}
harfler = set(['a', 'b', 'c'])

# Set işlemleri
sayilar.add(6)          # Eleman ekleme
sayilar.remove(1)       # Eleman silme
sayilar.discard(10)     # Güvenli silme (hata vermez)

# Küme işlemleri
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
birlesim = A | B        # Birleşim
kesisim = A & B        # Kesişim
fark = A - B           # Fark
\`\`\`

## 5. Array'ler

Array'ler, aynı veri tipindeki elemanları saklamak için kullanılır. NumPy kütüphanesi ile daha gelişmiş array işlemleri yapılabilir.

\`\`\`python
from array import array

# Array oluşturma
sayilar = array('i', [1, 2, 3, 4, 5])  # 'i': signed integer

# Array işlemleri
sayilar.append(6)       # Eleman ekleme
sayilar.extend([7, 8])  # Çoklu ekleme
sayilar.pop()          # Son elemanı çıkarma
sayilar.remove(3)      # Belirli elemanı çıkarma
\`\`\`

## 6. Stack ve Queue

Python listelerini kullanarak stack (yığın) ve queue (kuyruk) veri yapılarını implemente edebiliriz.

### Stack (LIFO - Last In First Out)

\`\`\`python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Kullanım
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 3
print(stack.peek()) # 2
\`\`\`

### Queue (FIFO - First In First Out)

\`\`\`python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Kullanım
queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # 1
print(queue.front())    # 2
\`\`\`

## Alıştırmalar

1. **Liste İşlemleri**
   - Verilen bir listedeki tekrar eden elemanları bulun
   - Listeyi tersine çevirin (slice kullanmadan)
   - İki listeyi birleştirip sıralayın

2. **Dictionary Uygulaması**
   - Öğrenci not sistemi oluşturun
   - Telefon rehberi uygulaması yapın
   - Alışveriş sepeti sistemi geliştirin

3. **Set Problemleri**
   - İki metin arasındaki ortak karakterleri bulun
   - Bir listedeki benzersiz elemanları bulun
   - Poker eli kategorileri belirleyin

## Kaynaklar

- [Python Resmi Dokümantasyonu - Veri Yapıları](https://docs.python.org/3/tutorial/datastructures.html)
- [Real Python - Python Veri Yapıları](https://realpython.com/python-data-structures/)
- [GeeksforGeeks - Python Veri Yapıları](https://www.geeksforgeeks.org/python-data-structures/)
`;

export default function BasicDataStructuresPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar">
            <ArrowLeft className="h-4 w-4" />
            Veri Yapıları ve Algoritmalara Dön
          </Link>
        </Button>
      </div>
      
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </div>
      
      {/* Interactive Examples */}
      <div className="my-12">
        <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
        <Tabs defaultValue="list">
          <TabsList>
            <TabsTrigger value="list">Liste</TabsTrigger>
            <TabsTrigger value="dict">Dictionary</TabsTrigger>
            <TabsTrigger value="set">Set</TabsTrigger>
          </TabsList>
          
          <TabsContent value="list">
            <Card>
              <CardHeader>
                <CardTitle>Liste İşlemleri</CardTitle>
                <CardDescription>
                  Python listelerinin temel kullanımı
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Liste oluşturma ve işlemler
sayilar = [1, 2, 3, 4, 5]

# Ekleme
sayilar.append(6)
print(sayilar)  # [1, 2, 3, 4, 5, 6]

# Çıkarma
sayilar.pop()
print(sayilar)  # [1, 2, 3, 4, 5]

# Dilimleme
print(sayilar[1:4])  # [2, 3, 4]`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="dict">
            <Card>
              <CardHeader>
                <CardTitle>Dictionary İşlemleri</CardTitle>
                <CardDescription>
                  Python dictionary'lerinin temel kullanımı
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Dictionary oluşturma
kisi = {
    "ad": "Ahmet",
    "yas": 25,
    "sehir": "İstanbul"
}

# Değer ekleme/güncelleme
kisi["meslek"] = "Mühendis"
print(kisi)

# Değer alma
print(kisi.get("ad"))  # Ahmet
print(kisi.get("tel", "Bulunamadı"))  # Bulunamadı`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="set">
            <Card>
              <CardHeader>
                <CardTitle>Set İşlemleri</CardTitle>
                <CardDescription>
                  Python set'lerinin temel kullanımı
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# Set oluşturma
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

# Küme işlemleri
print(A | B)  # Birleşim: {1, 2, 3, 4, 5, 6}
print(A & B)  # Kesişim: {3, 4}
print(A - B)  # Fark: {1, 2}
print(A ^ B)  # Simetrik fark: {1, 2, 5, 6}`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button variant="outline" disabled className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Önceki Konu
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/ileri-veri-yapilari">
            Sonraki Konu: İleri Veri Yapıları
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 