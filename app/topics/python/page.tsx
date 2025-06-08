'use client';

import Link from 'next/link';
import Image from 'next/image';
import { Code2, Database, Brain, Globe, ChartBar, Terminal, Puzzle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import MarkdownContent from '@/components/MarkdownContent';

const content = `
# Python Eğitimleri

Python, basit ve okunabilir sözdizimi, zengin kütüphane ekosistemi ve geniş kullanım alanlarıyla dünyanın en popüler programlama dillerinden biridir. Yapay zeka, veri bilimi, web geliştirme ve otomasyon gibi birçok alanda tercih edilmektedir.

## Python ile Neler Yapabilirsiniz?

- **Yapay Zeka ve Makine Öğrenmesi**: TensorFlow, PyTorch ve scikit-learn ile yapay zeka modelleri geliştirin
- **Veri Bilimi ve Analizi**: Pandas, NumPy ve Matplotlib ile veri analizi ve görselleştirme yapın
- **Web Uygulamaları**: Django ve Flask ile modern web uygulamaları geliştirin
- **Otomasyon ve Scripting**: Tekrarlayan görevleri otomatikleştirin ve sistem yönetimi yapın
- **Bilimsel Hesaplama**: Karmaşık matematiksel işlemleri ve simülasyonları gerçekleştirin

## Neden Python Öğrenmeliyim?

### 1. Kolay Öğrenme Eğrisi
- Açık ve anlaşılır sözdizimi
- Zengin Türkçe kaynaklar
- Pratik örnekler ve projeler
- Adım adım öğrenme yolu

### 2. Geniş Kullanım Alanları
- Yapay zeka ve makine öğrenmesi
- Veri bilimi ve analizi
- Web geliştirme
- Sistem yönetimi ve otomasyon
- Oyun geliştirme

### 3. Güçlü Ekosistem
- 300,000+ hazır paket ve kütüphane
- Aktif geliştirici topluluğu
- Kapsamlı dokümantasyon
- Ücretsiz kaynaklar ve araçlar

### 4. Kariyer Fırsatları
- Yüksek maaş potansiyeli
- Artan iş imkanları
- Uzaktan çalışma fırsatları
- Freelance projeler

## Öğrenme Yolculuğunuz

Kodleon'un Python eğitim serisi, temel kavramlardan ileri seviye uygulamalara kadar kapsamlı bir öğrenme deneyimi sunar. Her bölüm, teorik bilgilerin yanı sıra pratik örnekler ve alıştırmalar içerir.

Eğitimlerimiz, modern yazılım endüstrisinin ihtiyaçları doğrultusunda tasarlanmıştır ve sürekli güncellenmektedir. Yapay zeka çağında, Python programlama becerilerinizi geliştirerek geleceğe hazır olun.
`;

const learningPathItems = [
  { id: 1, title: "Python Temelleri", description: "Değişkenler, veri tipleri, operatörler, kontrol akışı ve fonksiyonlar." },
  { id: 2, title: "Nesne Tabanlı Programlama", description: "Sınıflar, nesneler, kalıtım ve çok biçimlilik kavramları." },
  { id: 3, title: "Veri Yapıları ve Algoritmalar", description: "Listeler, demetler, sözlükler, kümeler ve temel algoritmalar." },
  { id: 4, title: "İleri Python Özellikleri", description: "Jeneratörler, dekoratörler, lambda fonksiyonları ve dosya işlemleri." },
  { id: 5, title: "Uzmanlık Alanları", description: "Web Geliştirme (Django/Flask), Veri Bilimi (Pandas/NumPy), Yapay Zeka (TensorFlow/PyTorch)." }
];

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
    ],
    color: "from-blue-500/20 to-blue-500/10 hover:from-blue-500/30 hover:to-blue-500/20"
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
    ],
    color: "from-purple-500/20 to-purple-500/10 hover:from-purple-500/30 hover:to-purple-500/20"
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
    ],
    color: "from-green-500/20 to-green-500/10 hover:from-green-500/30 hover:to-green-500/20"
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
    ],
    color: "from-pink-500/20 to-pink-500/10 hover:from-pink-500/30 hover:to-pink-500/20"
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
    ],
    color: "from-orange-500/20 to-orange-500/10 hover:from-orange-500/30 hover:to-orange-500/20"
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
    ],
    color: "from-cyan-500/20 to-cyan-500/10 hover:from-cyan-500/30 hover:to-cyan-500/20"
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
    ],
    color: "from-yellow-500/20 to-yellow-500/10 hover:from-yellow-500/30 hover:to-yellow-500/20"
  }
];

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "Course",
  "name": "Python Eğitimleri",
  "description": "Python programlama dilini baştan sona öğrenin. Yapay zeka, veri bilimi, web geliştirme ve otomasyon alanlarında uzmanlaşın.",
  "provider": {
    "@type": "Organization",
    "name": "Kodleon",
    "sameAs": "https://kodleon.com"
  },
  "educationalLevel": "Beginner to Advanced",
  "courseCode": "PY-101",
  "hasCourseInstance": {
    "@type": "CourseInstance",
    "courseMode": "online",
    "inLanguage": "tr-TR"
  },
  "teaches": [
    "Python Programming",
    "Artificial Intelligence",
    "Data Science",
    "Web Development",
    "Automation"
  ],
  "learningResourceType": "Course",
  "audience": {
    "@type": "Audience",
    "audienceType": "Beginners, Intermediate Developers, Data Scientists, AI Engineers"
  }
};

export default function PythonTopicsPage() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="prose prose-lg dark:prose-invert mb-12">
            <MarkdownContent content={content} />
          </div>

          {/* Learning Path Section */}
          <div className="mb-16">
            <h2 className="text-2xl md:text-3xl font-semibold tracking-tight mb-8 text-center">Python Öğrenme Yolu</h2>
            <div className="relative">
              {/* Optional: Timeline line */}
              {/* <div className="hidden md:block absolute top-0 left-1/2 w-px h-full bg-border -translate-x-1/2"></div> */}
              
              <div className="space-y-0 md:grid md:grid-cols-1 md:gap-x-8 relative">
                {learningPathItems.map((item, index) => (
                  <div key={item.id} className="relative flex items-start space-x-4 pb-8">
                    {/* Connecting line */}
                    {index !== learningPathItems.length - 1 && (
                      <div className="absolute left-6 top-12 bottom-0 w-0.5 bg-border ml-px"></div>
                    )}
                    <div className="flex-shrink-0 w-12 h-12 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold text-lg shadow-md z-10">
                      {item.id}
                    </div>
                    <div className="flex-1 pt-1.5 md:pt-2.5">
                      <h3 className="text-lg font-semibold text-foreground mb-1">{item.title}</h3>
                      <p className="text-sm text-muted-foreground">{item.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {topics.map((topic, index) => {
              const Icon = topic.icon;
              return (
                <Card 
                  key={index} 
                  className={`flex flex-col bg-gradient-to-br transition-all duration-300 ${topic.color} border-none shadow-lg`}
                >
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Icon className="h-6 w-6" />
                      <CardTitle>{topic.title}</CardTitle>
                    </div>
                    <CardDescription className="text-foreground/80">{topic.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex-1">
                    <ul className="list-disc list-inside space-y-1 text-sm text-foreground/70">
                      {topic.features.map((feature, i) => (
                        <li key={i}>{feature}</li>
                      ))}
                    </ul>
                  </CardContent>
                  <div className="p-6 pt-0">
                    <Button 
                      asChild 
                      className="w-full bg-background/50 hover:bg-background/70 text-foreground border border-border"
                    >
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
    </>
  );
} 