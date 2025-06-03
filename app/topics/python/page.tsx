import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { Code2, Database, Brain, Globe, ChartBar, Terminal, Puzzle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python Eğitimleri | Kodleon',
  description: 'Python programlama dilini temellerden ileri seviyeye kadar öğrenin. Veri yapıları, algoritmalar, web geliştirme, veri bilimi ve daha fazlası.',
};

const content = `
# Python Eğitimleri

Python, basit ve okunabilir sözdizimi, zengin kütüphane ekosistemi ve geniş kullanım alanlarıyla dünyanın en popüler programlama dillerinden biridir. Yapay zeka, veri bilimi, web geliştirme ve otomasyon gibi birçok alanda tercih edilmektedir.

## Neden Python Öğrenmeliyim?

- **Kolay Öğrenme**: Açık ve anlaşılır sözdizimi
- **Geniş Kullanım**: Veri bilimi, yapay zeka, web geliştirme ve daha fazlası
- **Zengin Ekosistem**: 300,000+ paket ve kütüphane
- **Güçlü Topluluk**: Aktif geliştirici topluluğu ve kaynaklar
- **Yüksek Verimlilik**: Hızlı geliştirme ve prototipleme

## Öğrenme Yolu

1. Python Temelleri
2. Nesne Tabanlı Programlama
3. Veri Yapıları ve Algoritmalar
4. İleri Python Özellikleri
5. Uzmanlık Alanları (Web, Veri Bilimi, Yapay Zeka)
`;

const topics = [
  {
    title: "Python Temelleri",
    href: "/topics/python/temel-python",
    description: "Python'un temel kavramlarını öğrenin: değişkenler, veri tipleri, kontrol yapıları, fonksiyonlar ve nesne yönelimli programlama.",
    icon: Code2,
    features: [
      "Değişkenler ve Veri Tipleri",
      "Kontrol Yapıları",
      "Fonksiyonlar",
      "Temel OOP Kavramları",
      "Modüller ve Paketler"
    ]
  },
  {
    title: "Nesne Tabanlı Programlama",
    href: "/topics/python/nesne-tabanli-programlama",
    description: "Python'da nesne tabanlı programlamanın (OOP) tüm detaylarını öğrenin: sınıflar, kalıtım, kapsülleme ve çok biçimlilik.",
    icon: Puzzle,
    features: [
      "Sınıflar ve Nesneler",
      "Kalıtım ve Hiyerarşi",
      "Kapsülleme",
      "Çok Biçimlilik",
      "İleri OOP Teknikleri"
    ]
  },
  {
    title: "Veri Yapıları ve Algoritmalar",
    href: "/topics/python/veri-yapilari-ve-algoritmalar",
    description: "Temel veri yapıları ve algoritmaları Python ile öğrenin. Listeler, ağaçlar, graflar ve sıralama algoritmaları.",
    icon: Database,
    features: [
      "Temel Veri Yapıları",
      "Arama Algoritmaları",
      "Sıralama Algoritmaları",
      "Graflar ve Ağaçlar",
      "Algoritma Analizi"
    ]
  },
  {
    title: "Derin Öğrenme",
    href: "/topics/python/derin-ogrenme",
    description: "Yapay sinir ağları, derin öğrenme modelleri ve uygulamaları hakkında kapsamlı eğitim.",
    icon: Brain,
    features: [
      "Yapay Sinir Ağları",
      "Derin Öğrenme Frameworkleri",
      "CNN ve RNN",
      "Transfer Öğrenme",
      "Model Optimizasyonu"
    ]
  },
  {
    title: "Web Geliştirme",
    href: "/topics/python/web-gelistirme",
    description: "Python web frameworkleri ile modern web uygulamaları geliştirmeyi öğrenin.",
    icon: Globe,
    features: [
      "Flask ve Django",
      "REST API Geliştirme",
      "Veritabanı Entegrasyonu",
      "Web Güvenliği",
      "Deployment"
    ]
  },
  {
    title: "Veri Bilimi",
    href: "/topics/python/veri-bilimi",
    description: "Veri analizi, görselleştirme ve makine öğrenmesi ile veri bilimi uygulamaları.",
    icon: ChartBar,
    features: [
      "Pandas ve NumPy",
      "Veri Görselleştirme",
      "İstatistiksel Analiz",
      "Makine Öğrenmesi",
      "Büyük Veri İşleme"
    ]
  },
  {
    title: "Otomasyon ve Scripting",
    href: "/topics/python/otomasyon",
    description: "Sistem yönetimi, otomasyon ve scripting ile tekrarlayan görevleri otomatikleştirin.",
    icon: Terminal,
    features: [
      "Dosya İşlemleri",
      "Web Scraping",
      "Task Otomasyonu",
      "Sistem Yönetimi",
      "GUI Otomasyon"
    ]
  }
];

export default function PythonTopicsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="prose prose-lg dark:prose-invert mb-12">
          <MarkdownContent content={content} />
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {topics.map((topic, index) => {
            const Icon = topic.icon;
            return (
              <Card key={index} className="flex flex-col">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Icon className="h-6 w-6" />
                    <CardTitle>{topic.title}</CardTitle>
                  </div>
                  <CardDescription>{topic.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-1">
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {topic.features.map((feature, i) => (
                      <li key={i}>{feature}</li>
                    ))}
                  </ul>
                </CardContent>
                <div className="p-6 pt-0">
                  <Button asChild className="w-full">
                    <Link href={topic.href}>Öğrenmeye Başla</Link>
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 