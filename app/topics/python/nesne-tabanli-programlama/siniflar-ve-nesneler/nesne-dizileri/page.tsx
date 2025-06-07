import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, List, Terminal, BookOpen } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Nesne Dizileri | AIKOD',
  description: 'Python\'da nesne dizileriyle çalışma, liste ve tuple kullanımı, döngülerle işlem yapma ve toplu nesne işlemleri.',
};

const content = `
# Python'da Nesne Dizileriyle Çalışma

Nesne tabanlı programlamada, birden fazla nesneyi yönetmek ve üzerlerinde işlem yapmak için nesne dizilerini kullanırız. Bu bölümde, nesne dizileriyle çalışmanın temellerini ve ileri seviye tekniklerini öğreneceğiz.

## Liste ve Tuple Kullanımı

Nesneleri saklamak için en yaygın kullanılan veri yapıları listeler ve tuple'lardır:

\`\`\`python
class Kitap:
    def __init__(self, baslik, yazar, sayfa_sayisi):
        self.baslik = baslik
        self.yazar = yazar
        self.sayfa_sayisi = sayfa_sayisi
    
    def __str__(self):
        return f"{self.baslik} - {self.yazar}"

# Liste kullanımı
kitaplar = [
    Kitap("Python Programlama", "Ahmet Yılmaz", 300),
    Kitap("Veri Bilimi", "Ayşe Demir", 450),
    Kitap("Yapay Zeka", "Mehmet Kaya", 400)
]

# Tuple kullanımı (değiştirilemez)
favori_kitaplar = (
    Kitap("Algoritma Temelleri", "Can Öztürk", 250),
    Kitap("Web Geliştirme", "Zeynep Şahin", 350)
)
\`\`\`

## Döngülerle İşlem Yapma

Nesne dizileri üzerinde çeşitli döngü yapıları kullanarak işlem yapabiliriz:

### For Döngüsü

\`\`\`python
# Tüm kitapları listele
for kitap in kitaplar:
    print(f"Kitap: {kitap.baslik}, Yazar: {kitap.yazar}")

# Sayfa sayısı 300'den fazla olan kitapları filtrele
uzun_kitaplar = [kitap for kitap in kitaplar if kitap.sayfa_sayisi > 300]
\`\`\`

### Enumerate Kullanımı

\`\`\`python
# İndeks ile birlikte listeleme
for i, kitap in enumerate(kitaplar, 1):
    print(f"{i}. {kitap.baslik}")
\`\`\`

## Toplu Nesne İşlemleri

Nesne dizileri üzerinde çeşitli toplu işlemler gerçekleştirebiliriz:

### Filtreleme

\`\`\`python
class Ogrenci:
    def __init__(self, ad, not_ortalamasi):
        self.ad = ad
        self.not_ortalamasi = not_ortalamasi

ogrenciler = [
    Ogrenci("Ali", 85),
    Ogrenci("Veli", 75),
    Ogrenci("Ayşe", 90),
    Ogrenci("Fatma", 95)
]

# Başarılı öğrencileri filtrele
basarili_ogrenciler = list(filter(lambda x: x.not_ortalamasi >= 85, ogrenciler))

# List comprehension ile filtreleme
basarili_ogrenciler = [o for o in ogrenciler if o.not_ortalamasi >= 85]
\`\`\`

### Sıralama

\`\`\`python
# Not ortalamasına göre sırala
ogrenciler.sort(key=lambda x: x.not_ortalamasi, reverse=True)

# sorted() fonksiyonu ile sıralama
sirali_ogrenciler = sorted(ogrenciler, key=lambda x: x.not_ortalamasi)
\`\`\`

### Dönüştürme (Map)

\`\`\`python
# Tüm öğrencilerin notlarını 5 puan artır
def not_artir(ogrenci):
    ogrenci.not_ortalamasi += 5
    return ogrenci

guncel_ogrenciler = list(map(not_artir, ogrenciler))
\`\`\`

## Nesne Koleksiyonları

Python'da nesneleri yönetmek için özel koleksiyon türleri de kullanabiliriz:

### Set Kullanımı

\`\`\`python
# Benzersiz nesneler için set kullanımı
class Kullanici:
    def __init__(self, id, ad):
        self.id = id
        self.ad = ad
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

# Benzersiz kullanıcılar kümesi
kullanicilar = {
    Kullanici(1, "Ali"),
    Kullanici(2, "Veli"),
    Kullanici(1, "Ali")  # Aynı ID'ye sahip, eklenmeyecek
}
\`\`\`

### Dictionary Kullanımı

\`\`\`python
# Nesneleri anahtar-değer çiftleri olarak saklama
ogrenci_dict = {
    "123": Ogrenci("Ali", 85),
    "124": Ogrenci("Veli", 75),
    "125": Ogrenci("Ayşe", 90)
}

# ID'ye göre öğrenci bulma
print(ogrenci_dict["123"].ad)  # Ali
\`\`\`

## İyi Pratikler

1. **Tip Kontrolü**: Dizilerde aynı türden nesneleri saklayın
2. **Bellek Yönetimi**: Çok büyük nesne dizileri için generator kullanmayı düşünün
3. **Performans**: Sık erişilen nesneler için dictionary kullanın
4. **Okunabilirlik**: List comprehension'ları karmaşık hale getirmeyin
5. **Güvenlik**: Nesne dizilerini değiştirirken dikkatli olun

## Yaygın Hatalar ve Çözümleri

1. **Shallow vs Deep Copy**:
\`\`\`python
import copy

# Shallow copy - sadece referansları kopyalar
shallow_list = ogrenciler.copy()

# Deep copy - nesnelerin kendisini kopyalar
deep_list = copy.deepcopy(ogrenciler)
\`\`\`

2. **None Kontrolü**:
\`\`\`python
# Güvenli erişim için None kontrolü
for ogrenci in ogrenciler:
    if ogrenci is not None and hasattr(ogrenci, 'not_ortalamasi'):
        print(ogrenci.not_ortalamasi)
\`\`\`

3. **Iterator Tüketimi**:
\`\`\`python
# Iterator'ı listeye dönüştürerek birden fazla kez kullanma
filtered_iter = filter(lambda x: x.not_ortalamasi >= 85, ogrenciler)
filtered_list = list(filtered_iter)  # Iterator'ı listeye dönüştür
\`\`\`
`;

const sections = [
  {
    title: "Temel Kullanım",
    description: "Liste ve tuple ile nesne saklama",
    icon: <List className="h-6 w-6" />,
    topics: [
      "Liste oluşturma",
      "Tuple kullanımı",
      "Nesne ekleme",
      "Erişim yöntemleri"
    ]
  },
  {
    title: "Döngü İşlemleri",
    description: "Nesne dizileri üzerinde iterasyon",
    icon: <Terminal className="h-6 w-6" />,
    topics: [
      "For döngüsü",
      "Enumerate kullanımı",
      "List comprehension",
      "Generator expressions"
    ]
  },
  {
    title: "Toplu İşlemler",
    description: "Filtreleme, sıralama ve dönüştürme",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Filter fonksiyonu",
      "Sort metodları",
      "Map kullanımı",
      "Lambda ifadeleri"
    ]
  },
  {
    title: "İleri Teknikler",
    description: "Özel koleksiyon yapıları",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "Set kullanımı",
      "Dictionary yapısı",
      "Custom iterators",
      "Context managers"
    ]
  }
];

export default function NesneDizileriPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/50 dark:hover:bg-blue-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-blue-600 dark:text-blue-400">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription className="dark:text-gray-300">{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                    {section.topics.map((topic, i) => (
                      <li key={i}>{topic}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Back Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler">
              Sınıflar ve Nesneler Sayfasına Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 