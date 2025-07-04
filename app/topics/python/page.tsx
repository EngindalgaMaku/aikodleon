'use client';

import Link from 'next/link';
import Image from 'next/image';
import { Code2, Database, Brain, Globe, ChartBar, Terminal, Puzzle, Sparkles, Bot, Cpu } from 'lucide-react';
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
    color: "from-blue-500/20 to-blue-500/10 hover:from-blue-500/30 hover:to-blue-500/20",
    iconBg: "bg-blue-500/10 text-blue-600"
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
    color: "from-purple-500/20 to-purple-500/10 hover:from-purple-500/30 hover:to-purple-500/20",
    iconBg: "bg-purple-500/10 text-purple-600"
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
    color: "from-green-500/20 to-green-500/10 hover:from-green-500/30 hover:to-green-500/20",
    iconBg: "bg-green-500/10 text-green-600"
  },
  {
    title: "Derin Öğrenme",
    href: "/topics/python/derin-ogrenme",
    description: "Yapay sinir ağları, derin öğrenme modelleri ve uygulamaları hakkında kapsamlı eğitim.",
    icon: Brain,
    features: [
      { text: "Yapay Sinir Ağları", href: "/kod-ornekleri/temel-sinir-agi" },
      { text: "PyTorch Dersleri", href: "/topics/python/pytorch-dersleri/01-pytorch-kurulumu-ve-tensorlere-giris" },
      { text: "Derin Öğrenme Frameworkleri" },
      { text: "CNN ve RNN" },
      { text: "Transfer Öğrenme" },
      { text: "Model Optimizasyonu" }
    ],
    color: "from-pink-500/20 to-pink-500/10 hover:from-pink-500/30 hover:to-pink-500/20",
    iconBg: "bg-pink-500/10 text-pink-600"
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
    color: "from-orange-500/20 to-orange-500/10 hover:from-orange-500/30 hover:to-orange-500/20",
    iconBg: "bg-orange-500/10 text-orange-600"
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
    color: "from-cyan-500/20 to-cyan-500/10 hover:from-cyan-500/30 hover:to-cyan-500/20",
    iconBg: "bg-cyan-500/10 text-cyan-600"
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
    color: "from-yellow-500/20 to-yellow-500/10 hover:from-yellow-500/30 hover:to-yellow-500/20",
    iconBg: "bg-yellow-500/10 text-yellow-600"
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
      
      {/* AI-themed background pattern */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-0 w-full h-full bg-grid-pattern opacity-5"></div>
        <div className="absolute top-40 right-20 w-96 h-96 rounded-full bg-blue-500/5 blur-3xl"></div>
        <div className="absolute top-80 left-20 w-72 h-72 rounded-full bg-purple-500/5 blur-3xl"></div>
        <div className="absolute bottom-40 right-40 w-80 h-80 rounded-full bg-cyan-500/5 blur-3xl"></div>
      </div>

      <div className="container mx-auto px-4 py-8 relative z-10">
        <div className="max-w-6xl mx-auto">
          {/* Hero section with AI visual elements */}
          <div className="relative mb-12 p-6 rounded-2xl bg-gradient-to-br from-blue-50/50 to-purple-50/50 dark:from-blue-950/30 dark:to-purple-950/30 border border-blue-100/50 dark:border-blue-800/30 shadow-lg overflow-hidden">
            <div className="absolute top-0 right-0 -mt-10 -mr-10">
              <div className="w-40 h-40 bg-blue-500/10 rounded-full blur-2xl"></div>
            </div>
            <div className="absolute bottom-0 left-0 -mb-10 -ml-10">
              <div className="w-40 h-40 bg-purple-500/10 rounded-full blur-2xl"></div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-8 items-center">
              <div className="flex-1">
                <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 text-sm font-medium mb-4">
                  <Cpu className="h-4 w-4 mr-2" />
                  Yapay Zeka & Python
                </div>
                <h1 className="text-3xl md:text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
                  Python ile Yapay Zeka Dünyasına Adım Atın
                </h1>
                <div className="prose prose-lg dark:prose-invert mb-6">
                  <p className="text-gray-700 dark:text-gray-300">
                    Python, yapay zeka ve veri bilimi alanında en çok tercih edilen programlama dilidir. 
                    Kodleon'un kapsamlı Python eğitimleriyle, yapay zeka uygulamaları geliştirmeye hemen başlayın.
                  </p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <div className="inline-flex items-center px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-sm">
                    <Sparkles className="h-4 w-4 mr-1 text-yellow-500" />
                    Yapay Zeka
                  </div>
                  <div className="inline-flex items-center px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-sm">
                    <Brain className="h-4 w-4 mr-1 text-purple-500" />
                    Derin Öğrenme
                  </div>
                  <div className="inline-flex items-center px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-sm">
                    <Bot className="h-4 w-4 mr-1 text-blue-500" />
                    Makine Öğrenmesi
                  </div>
                  <div className="inline-flex items-center px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-sm">
                    <Database className="h-4 w-4 mr-1 text-green-500" />
                    Veri Bilimi
                  </div>
                </div>
              </div>
              <div className="flex-shrink-0 w-full md:w-1/3 relative">
                <div className="aspect-square relative">
                  <Image 
                    src="/images/python-ai-illustration.jpg" 
                    alt="Python ve Yapay Zeka" 
                    width={300}
                    height={300}
                    className="rounded-lg shadow-lg object-cover"
                    onError={(e) => {
                      e.currentTarget.src = "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg";
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Course Cards Section - Moved up */}
          <div className="mb-16">
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-8 text-center relative">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
                Python & AI Kurs Konuları
              </span>
              <div className="absolute w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 bottom-0 left-1/2 transform -translate-x-1/2 mt-2 rounded-full"></div>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {topics.map((topic) => (
                <Link key={topic.title} href={topic.href} className="flex">
                  <Card className={`w-full flex flex-col bg-gradient-to-br transition-all duration-300 ${topic.color} border-gray-200 dark:border-gray-800 shadow-md hover:shadow-xl hover:-translate-y-1`}>
                    <CardHeader>
                      <div className="flex items-center gap-4">
                        <div className={`p-3 rounded-lg ${topic.iconBg}`}>
                          <topic.icon className="h-6 w-6" />
                        </div>
                        <CardTitle className="text-xl font-bold">{topic.title}</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="flex-grow">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">{topic.description}</p>
                      <ul className="space-y-2 text-sm">
                        {topic.features.map((feature, index) => (
                          <li key={index} className="flex items-center">
                            <div className="w-2 h-2 bg-current rounded-full mr-3 flex-shrink-0"></div>
                            {typeof feature === "object" && feature !== null && "href" in feature ? (
                              <a href={feature.href} className="hover:underline text-blue-600 dark:text-blue-400 font-semibold">
                                {feature.text}
                              </a>
                            ) : (
                              typeof feature === "object" && feature !== null && "text" in feature
                                ? feature.text
                                : feature
                            )}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                    <div className="p-4 mt-auto">
                      <Button variant="ghost" className="w-full bg-white/50 dark:bg-black/20 hover:bg-white/70 dark:hover:bg-black/30">
                        Öğrenmeye Başla <Sparkles className="h-4 w-4 ml-2" />
                      </Button>
                    </div>
                  </Card>
                </Link>
              ))}
            </div>
          </div>

          {/* Learning Path Section - Moved down with AI theme */}
          <div className="mb-16 relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-50/30 to-purple-50/30 dark:from-blue-950/20 dark:to-purple-950/20 rounded-2xl -z-10"></div>
            <div className="p-8 rounded-2xl border border-blue-100/50 dark:border-blue-800/30">
              <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-8 text-center">
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
                  Python Öğrenme Yolu
                </span>
              </h2>
              <div className="relative">
                <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-400 to-purple-500 dark:from-blue-500 dark:to-purple-600"></div>
                <div className="space-y-0 relative">
                  {learningPathItems.map((item, index) => (
                    <div key={item.id} className="relative flex items-start space-x-4 pb-8">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-lg shadow-lg z-10">
                        {item.id}
                      </div>
                      <div className="flex-1 pt-1.5 md:pt-2.5 bg-white/50 dark:bg-gray-900/50 p-4 rounded-lg border border-white/20 dark:border-gray-800/50 shadow-sm">
                        <h3 className="text-lg font-semibold text-foreground mb-1">{item.title}</h3>
                        <p className="text-sm text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-16 text-center text-sm text-muted-foreground">
            <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
          </div>
        </div>
      </div>
    </>
  );
} 