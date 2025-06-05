import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, BookOpen, Code2, FileCode2, Lightbulb, Puzzle, Trophy, Box, GitFork, Lock, Layers, Component } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python ile Nesneye Yönelik Programlama',
  description: 'Python\'da nesneye yönelik programlamanın (OOP) temelleri, sınıflar, nesneler, kalıtım ve daha fazlası.',
};

const content = `
# Python ile Nesneye Yönelik Programlama (OOP)

Python'da nesneye yönelik programlama (OOP), kodunuzu daha modüler, okunabilir ve yeniden kullanılabilir hale getiren güçlü bir programlama paradigmasıdır. Bu rehberde, OOP'nin temel kavramlarını ve Python'da nasıl uygulandığını detaylı örneklerle öğreneceksiniz.

## Neden OOP Öğrenmeliyiz?

Nesneye yönelik programlama, modern yazılım geliştirmenin temel taşlarından biridir. OOP ile:

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

## Öğrenme Yolculuğunuz

Bu eğitim serisi, başlangıç seviyesinden ileri seviyeye kadar OOP kavramlarını kapsar. Her bölüm, teorik bilgilerin yanı sıra pratik örnekler ve alıştırmalar içerir.
`;

const sections = [
  {
    title: "1. Sınıflar ve Nesneler",
    description: "OOP'nin temel yapı taşları olan sınıf ve nesne kavramlarını öğrenin.",
    image: "/images/python-oop/classes-objects.jpg",
    icon: <Code2 className="h-6 w-6" />,
    href: "/topics/python/nesneye-yonelik-programlama/siniflar-ve-nesneler",
    topics: [
      "Sınıf nedir?",
      "Nesne oluşturma",
      "Constructor (__init__)",
      "Instance metodları",
      "Self parametresi"
    ]
  },
  {
    title: "2. Kalıtım",
    description: "Sınıflar arası kalıtım ve kod yeniden kullanımı tekniklerini keşfedin.",
    image: "/images/python-oop/inheritance.jpg",
    icon: <FileCode2 className="h-6 w-6" />,
    href: "/topics/python/nesneye-yonelik-programlama/kalitim",
    topics: [
      "Temel kalıtım",
      "Çoklu kalıtım",
      "Method overriding",
      "super() kullanımı",
      "Mixin sınıfları"
    ]
  },
  {
    title: "3. Kapsülleme",
    description: "Veri gizleme ve güvenli erişim yöntemlerini öğrenin.",
    image: "/images/python-oop/encapsulation.jpg",
    icon: <Puzzle className="h-6 w-6" />,
    href: "/topics/python/nesneye-yonelik-programlama/kapsulleme",
    topics: [
      "Private ve protected üyeler",
      "Getter ve setter metodları",
      "Property dekoratörü",
      "Name mangling",
      "Access modifiers"
    ]
  },
  {
    title: "4. Çok Biçimlilik",
    description: "Aynı arayüzü farklı sınıflarda kullanma tekniklerini keşfedin.",
    image: "/images/python-oop/polymorphism.jpg",
    icon: <Lightbulb className="h-6 w-6" />,
    href: "/topics/python/nesneye-yonelik-programlama/cok-bicimlilk",
    topics: [
      "Method overriding",
      "Duck typing",
      "Abstract base classes",
      "Interface tanımlama",
      "Polymorphic functions"
    ]
  },
  {
    title: "5. Soyut Sınıflar ve Arayüzler",
    description: "Soyut sınıflar ve arayüzler ile kod organizasyonunu öğrenin.",
    image: "/images/python-oop/abstract.jpg",
    icon: <Component className="h-6 w-6" />,
    href: "/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler",
    topics: [
      "Soyut sınıf nedir?",
      "ABC modülü",
      "Abstract metodlar",
      "Protokoller",
      "Interface tanımlama"
    ]
  },
  {
    title: "6. İleri Düzey Konular",
    description: "OOP'nin güçlü ve ileri seviye özelliklerini öğrenin.",
    image: "/images/python-oop/advanced.jpg",
    icon: <Trophy className="h-6 w-6" />,
    href: "/topics/python/nesneye-yonelik-programlama/ileri-duzey",
    topics: [
      "Magic methods",
      "Decorators",
      "Metaclasses",
      "Context managers",
      "Descriptors"
    ]
  }
];

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
              <Card key={index} className="group hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-primary/10">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription>{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {section.topics.map((topic, i) => (
                      <li key={i}>{topic}</li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button asChild className="w-full group">
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
            <Card>
              <CardHeader>
                <CardTitle>Video Eğitimler</CardTitle>
                <CardDescription>OOP kavramlarını görsel olarak öğrenin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Temel OOP Kavramları</li>
                  <li>Pratik Uygulamalar</li>
                  <li>İleri Seviye Teknikler</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Alıştırmalar</CardTitle>
                <CardDescription>Öğrendiklerinizi pekiştirin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Kod Örnekleri</li>
                  <li>Quiz Soruları</li>
                  <li>Projeler</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Dokümantasyon</CardTitle>
                <CardDescription>Detaylı kaynak dökümanlar</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Python OOP Rehberi</li>
                  <li>Best Practices</li>
                  <li>Tasarım Desenleri</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
} 