import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, BookOpen, Code2, FileCode2, Lightbulb, Puzzle, Trophy, Box, GitFork, Lock, Layers, Component, GitBranch, Shapes, Factory } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python ile Nesne Tabanlı Programlama',
  description: 'Python\'da nesne tabanlı programlamanın (OOP) temelleri, sınıflar, nesneler, kalıtım ve daha fazlası.',
};

const content = `
# Python ile Nesne Tabanlı Programlama (OOP)

Python'da nesne tabanlı programlama (OOP), kodunuzu daha modüler, okunabilir ve yeniden kullanılabilir hale getiren güçlü bir programlama paradigmasıdır. Bu rehberde, OOP'nin temel kavramlarını ve Python'da nasıl uygulandığını detaylı örneklerle öğreneceksiniz.

## Neden OOP Öğrenmeliyiz?

Nesne tabanlı programlama, modern yazılım geliştirmenin temel taşlarından biridir. OOP ile:

- Kodunuzu daha organize ve yönetilebilir hale getirebilirsiniz
- Kod tekrarını azaltabilirsiniz
- Büyük projeleri daha kolay yönetebilirsiniz
- Ekip çalışmasını kolaylaştırabilirsiniz
- Kodunuzu daha kolay test edebilirsiniz

## Temel OOP Kavramları

1. **Sınıflar ve Nesneler**: Kodunuzun yapı taşları
2. **Kalıtım**: Kod yeniden kullanımı ve hiyerarşi
3. **Kapsülleme**: Veri güvenliği ve gizlilik
4. **Çok Biçimlilik**: Esneklik ve genişletilebilirlik
5. **Soyut Sınıflar ve Arayüzler**: Kod organizasyonu ve standartlar

## Öğrenme Yolculuğunuz

Bu eğitim serisi, başlangıç seviyesinden ileri seviyeye kadar OOP kavramlarını kapsar. Her bölüm, teorik bilgilerin yanı sıra pratik örnekler ve alıştırmalar içerir.
`;

interface Section {
  title: string;
  description: string;
  image: string;
  icon: JSX.Element;
  href: string;
  topics: string[];
}

const sections: Section[] = [
  {
    title: "1. Sınıflar ve Nesneler",
    description: "Python'da sınıf ve nesne kavramları",
    image: "/images/python-oop/classes-objects.jpg",
    icon: <Code2 className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler",
    topics: [
      "Sınıf tanımlama",
      "Nesne oluşturma",
      "Metod yazma",
      "Özellik yönetimi"
    ]
  },
  {
    title: "2. Kalıtım",
    description: "Sınıflar arası kalıtım ilişkileri",
    image: "/images/python-oop/inheritance.jpg",
    icon: <GitBranch className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/kalitim",
    topics: [
      "Temel ve türetilmiş sınıflar",
      "super() kullanımı",
      "Çoklu kalıtım",
      "Method overriding"
    ]
  },
  {
    title: "3. Kapsülleme",
    description: "Veri gizleme ve erişim kontrolü",
    image: "/images/python-oop/encapsulation.jpg",
    icon: <Lock className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/kapsulleme",
    topics: [
      "Private değişkenler",
      "Getter ve setter metodları",
      "Property dekoratörü",
      "Name mangling"
    ]
  },
  {
    title: "4. Çok Biçimlilik",
    description: "Nesnelerin çok biçimli davranışı",
    image: "/images/python-oop/polymorphism.jpg",
    icon: <Shapes className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/cok-bicimlilk",
    topics: [
      "Method overriding",
      "Duck typing",
      "Abstract classes",
      "Interfaces"
    ]
  },
  {
    title: "5. Soyut Sınıflar ve Arayüzler",
    description: "Soyut sınıflar ve arayüz tanımları",
    image: "/images/python-oop/abstract.jpg",
    icon: <Component className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/soyut-siniflar-ve-arayuzler",
    topics: [
      "ABC modülü",
      "Soyut metodlar",
      "Interface tanımları",
      "Protocol sınıfları"
    ]
  },
  {
    title: "6. Tasarım Desenleri",
    description: "OOP tasarım desenleri ve kullanımları",
    image: "/images/python-oop/design-patterns.jpg",
    icon: <Factory className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/tasarim-desenleri",
    topics: [
      "Creational patterns",
      "Structural patterns",
      "Behavioral patterns",
      "Anti-patterns"
    ]
  },
  {
    title: "7. Pratik Örnekler",
    description: "Gerçek dünya OOP örnekleri",
    image: "/images/python-oop/advanced.jpg",
    icon: <BookOpen className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/pratik-ornekler",
    topics: [
      "Kütüphane sistemi",
      "Banka uygulaması",
      "E-ticaret sistemi",
      "Oyun geliştirme"
    ]
  }
];

const cardStyles: { [key: string]: string } = {
  "1. Sınıflar ve Nesneler": "bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/50 dark:hover:bg-blue-950/70",
  "2. Kalıtım": "bg-green-50 hover:bg-green-100 dark:bg-green-950/50 dark:hover:bg-green-950/70",
  "3. Kapsülleme": "bg-yellow-50 hover:bg-yellow-100 dark:bg-yellow-950/50 dark:hover:bg-yellow-950/70",
  "4. Çok Biçimlilik": "bg-purple-50 hover:bg-purple-100 dark:bg-purple-950/50 dark:hover:bg-purple-950/70",
  "5. Soyut Sınıflar ve Arayüzler": "bg-pink-50 hover:bg-pink-100 dark:bg-pink-950/50 dark:hover:bg-pink-950/70",
  "6. Tasarım Desenleri": "bg-orange-50 hover:bg-orange-100 dark:bg-orange-950/50 dark:hover:bg-orange-950/70",
  "7. Pratik Örnekler": "bg-teal-50 hover:bg-teal-100 dark:bg-teal-950/50 dark:hover:bg-teal-950/70"
};

const iconStyles: { [key: string]: string } = {
  "1. Sınıflar ve Nesneler": "text-blue-600 dark:text-blue-400",
  "2. Kalıtım": "text-green-600 dark:text-green-400",
  "3. Kapsülleme": "text-yellow-600 dark:text-yellow-400",
  "4. Çok Biçimlilik": "text-purple-600 dark:text-purple-400",
  "5. Soyut Sınıflar ve Arayüzler": "text-pink-600 dark:text-pink-400",
  "6. Tasarım Desenleri": "text-orange-600 dark:text-orange-400",
  "7. Pratik Örnekler": "text-teal-600 dark:text-teal-400"
};

export default function PythonOOPPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* OOP Concept Image */}
        <div className="my-8 flex justify-center">
          <Image
            src="/images/python_nesne1.jpg"
            alt="Python OOP Concepts"
            width={800}
            height={450}
            className="rounded-lg shadow-lg"
          />
        </div>
        
        {/* Interactive Learning Path */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Öğrenme Yolu</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className={`group transition-all duration-300 ${cardStyles[section.title]}`}>
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className={`p-2 rounded-lg ${iconStyles[section.title]}`}>
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription className="dark:text-gray-300">{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {section.topics.map((topic, i) => (
                      <li key={i} className="dark:text-gray-400">{topic}</li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button asChild className="w-full group bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100">
                    <Link href={section.href}>
                      Derse Git
                      <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>

        {/* Additional Resources */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Ek Kaynaklar</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-950/50 dark:hover:bg-indigo-950/70 transition-all duration-300">
              <CardHeader>
                <CardTitle>Terimler Sözlüğü</CardTitle>
                <CardDescription className="dark:text-gray-300">OOP terimlerini öğrenin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                  <li>Temel Kavramlar</li>
                  <li>İleri Kavramlar</li>
                  <li>Python'a Özgü Terimler</li>
                </ul>
              </CardContent>
              <CardFooter>
                <Button asChild className="w-full group bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100">
                  <Link href="/topics/python/nesne-tabanli-programlama/terimler-sozlugu">
                    Sözlüğe Git
                    <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
            <Card className="bg-rose-50 hover:bg-rose-100 dark:bg-rose-950/50 dark:hover:bg-rose-950/70 transition-all duration-300">
              <CardHeader>
                <CardTitle>Video Eğitimler</CardTitle>
                <CardDescription className="dark:text-gray-300">OOP kavramlarını görsel olarak öğrenin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                  <li>Temel OOP Kavramları</li>
                  <li>Pratik Uygulamalar</li>
                  <li>İleri Seviye Teknikler</li>
                </ul>
              </CardContent>
            </Card>
            <Card className="bg-cyan-50 hover:bg-cyan-100 dark:bg-cyan-950/50 dark:hover:bg-cyan-950/70 transition-all duration-300">
              <CardHeader>
                <CardTitle>Alıştırmalar</CardTitle>
                <CardDescription className="dark:text-gray-300">Öğrendiklerinizi pekiştirin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                  <li>Kod Örnekleri</li>
                  <li>Quiz Soruları</li>
                  <li>Projeler</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
} 