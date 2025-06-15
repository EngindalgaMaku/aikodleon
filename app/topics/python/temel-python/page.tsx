import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, Code2, Terminal, Database, FileCode2, Puzzle, Lightbulb } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python Temelleri | Kodleon',
  description: 'Python programlama dilinin temel kavramlarını öğrenin: değişkenler, veri tipleri, kontrol yapıları, fonksiyonlar ve daha fazlası.',
};

const content = `
# Python Temelleri

Python programlama dilinin temel kavramlarını ve yapı taşlarını bu bölümde öğreneceksiniz. Python'un basit ve okunabilir sözdizimi, onu öğrenmek için ideal bir dil yapar.

## Neden Python?

- **Kolay Öğrenme**: Açık ve anlaşılır sözdizimi
- **Geniş Kullanım**: Veri bilimi, web geliştirme, otomasyon
- **Zengin Kütüphane**: 300,000+ hazır paket
- **Güçlü Topluluk**: Aktif geliştirici desteği
- **Platform Bağımsız**: Windows, macOS, Linux

## Öğrenme Yolculuğunuz

Bu eğitim serisi, Python'un temel kavramlarını adım adım öğrenmenizi sağlayacak. Her bölüm, teorik bilgilerin yanı sıra pratik örnekler ve alıştırmalar içerir.
`;

// Card background colors
const cardColors = [
  'bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border-blue-200 dark:border-blue-800',
  'bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-green-200 dark:border-green-800',
  'bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border-purple-200 dark:border-purple-800',
  'bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 border-amber-200 dark:border-amber-800',
  'bg-gradient-to-br from-rose-50 to-rose-100 dark:from-rose-900/20 dark:to-rose-800/20 border-rose-200 dark:border-rose-800',
  'bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-900/20 dark:to-teal-800/20 border-teal-200 dark:border-teal-800',
];

const sections = [
  {
    title: "1. Python'a Giriş",
    description: "Python'un temel kavramlarını ve kurulum adımlarını öğrenin.",
    icon: <Terminal className="h-6 w-6" />,
    href: "/topics/python/temel-python/pythona-giris",
    topics: [
      "Python Nedir?",
      "Kurulum ve Ortam Hazırlığı",
      "İlk Python Programı",
      "Python Interpreter",
      "IDE Seçimi ve Kullanımı"
    ]
  },
  {
    title: "2. Değişkenler ve Veri Tipleri",
    description: "Python'da veri tipleri ve değişken kullanımını keşfedin.",
    icon: <Code2 className="h-6 w-6" />,
    href: "/topics/python/temel-python/degiskenler-ve-veri-tipleri",
    topics: [
      "Sayısal Veri Tipleri",
      "String (Metin) İşlemleri",
      "Listeler ve Tuple'lar",
      "Dictionary ve Set'ler",
      "Tip Dönüşümleri"
    ]
  },
  {
    title: "3. Kontrol Yapıları",
    description: "Koşullu ifadeler ve döngülerle program akışını kontrol edin.",
    icon: <FileCode2 className="h-6 w-6" />,
    href: "/topics/python/temel-python/kontrol-yapilari",
    topics: [
      "if-elif-else İfadeleri",
      "for Döngüsü",
      "while Döngüsü",
      "break ve continue",
      "try-except Blokları"
    ]
  },
  {
    title: "4. Fonksiyonlar",
    description: "Kod tekrarını önlemek için fonksiyonları kullanmayı öğrenin.",
    icon: <Puzzle className="h-6 w-6" />,
    href: "/topics/python/temel-python/fonksiyonlar",
    topics: [
      "Fonksiyon Tanımlama",
      "Parametreler ve Argümanlar",
      "Return İfadesi",
      "Lambda Fonksiyonları",
      "Modüller ve Import"
    ]
  },
  {
    title: "5. Veri Yapıları",
    description: "Python'un yerleşik veri yapılarını detaylı öğrenin.",
    icon: <Database className="h-6 w-6" />,
    href: "/topics/python/temel-python/veri-yapilari",
    topics: [
      "Liste İşlemleri",
      "Tuple Kullanımı",
      "Dictionary Metodları",
      "Set Operasyonları",
      "Array ve ByteArray"
    ]
  },
  {
    title: "6. Karışık Örnekler",
    description: "Tüm konuları içeren alıştırmalarla bilginizi pekiştirin.",
    icon: <Lightbulb className="h-6 w-6" />,
    href: "/topics/python/temel-python/karisik-ornekler",
    topics: [
      "Sayı Tahmin Oyunu",
      "Basit Hesap Makinesi",
      "Metin Analizi Aracı",
      "Öğrenci Not Sistemi",
      "Ve daha fazlası..."
    ]
  }
];

export default function PythonBasicsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python Eğitimleri
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert mb-12">
          <MarkdownContent content={content} />
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {sections.map((section, index) => (
            <Card key={index} className={`flex flex-col ${cardColors[index % cardColors.length]} hover:shadow-lg transition-shadow duration-300`}>
                  <CardHeader>
                <div className="flex items-center gap-2">
                  {section.icon}
                    <CardTitle>{section.title}</CardTitle>
                </div>
                <CardDescription>{section.description}</CardDescription>
                  </CardHeader>
              <CardContent className="flex-1">
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  {section.topics.map((topic, i) => (
                    <li key={i}>{topic}</li>
                  ))}
                </ul>
                  </CardContent>
              <div className="p-6 pt-0">
                <Button asChild className="w-full">
                  <Link href={section.href}>Derse Gir</Link>
                </Button>
              </div>
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