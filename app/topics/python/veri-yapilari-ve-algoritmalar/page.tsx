import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Binary, Box, GitFork, List, Network, Search } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python ile Veri Yapıları ve Algoritmalar',
  description: 'Python programlama dilinde temel veri yapıları ve algoritmaları öğrenin. Listeler, ağaçlar, grafikler ve arama/sıralama algoritmaları hakkında detaylı bilgi edinin.',
};

const content = `
# Python ile Veri Yapıları ve Algoritmalar

Veri yapıları ve algoritmalar, etkili ve verimli programlar yazmanın temelidir. Bu eğitim serisinde, Python kullanarak temel veri yapılarını ve algoritmaları öğrenecek, bunları gerçek dünya problemlerinde nasıl uygulayacağınızı keşfedeceksiniz.

## Neden Veri Yapıları ve Algoritmaları Öğrenmeliyiz?

- **Performans Optimizasyonu**: Doğru veri yapısı ve algoritma seçimi, programınızın performansını önemli ölçüde artırır
- **Problem Çözme Becerileri**: Karmaşık problemleri daha etkili bir şekilde çözmeyi öğrenirsiniz
- **Kod Kalitesi**: Daha temiz, daha organize ve daha bakımı kolay kod yazarsınız
- **Teknik Mülakatlar**: Yazılım geliştirici pozisyonları için yapılan teknik mülakatlarda bu konular sıkça sorulur
- **Ölçeklenebilirlik**: Büyük veri setleriyle çalışırken optimum çözümler üretebilirsiniz

## Öğrenme Yolculuğunuz

Bu eğitim serisi, temel kavramlardan ileri düzey uygulamalara kadar geniş bir yelpazede konuları kapsar. Her bölüm, teorik bilgilerin yanı sıra pratik örnekler ve alıştırmalar içerir.
`;

const sections = [
  {
    title: "1. Temel Veri Yapıları",
    description: "Python'da yerleşik veri yapılarını ve kullanımlarını öğrenin.",
    icon: <List className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/temel-veri-yapilari",
    topics: [
      "Listeler ve Tuple'lar",
      "Dictionary ve Set'ler",
      "Array ve ByteArray",
      "Stack ve Queue",
      "Linked List"
    ]
  },
  {
    title: "2. İleri Veri Yapıları",
    description: "Karmaşık veri yapılarını ve uygulamalarını keşfedin.",
    icon: <Binary className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/ileri-veri-yapilari",
    topics: [
      "Ağaçlar (Trees)",
      "Heap ve Priority Queue",
      "Hash Table",
      "Graf (Graph)",
      "Trie"
    ]
  },
  {
    title: "3. Temel Algoritmalar",
    description: "Algoritmaların temellerini ve analiz yöntemlerini öğrenin.",
    icon: <GitFork className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/temel-algoritmalar",
    topics: [
      "Algoritma Analizi (Big O)",
      "Brute Force",
      "Recursion",
      "Divide and Conquer",
      "Greedy Algoritmalar"
    ]
  },
  {
    title: "4. Sıralama Algoritmaları",
    description: "Farklı sıralama algoritmalarını ve kullanım senaryolarını öğrenin.",
    icon: <Box className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/siralama-algoritmalari",
    topics: [
      "Bubble Sort",
      "Selection Sort",
      "Insertion Sort",
      "Merge Sort",
      "Quick Sort"
    ]
  },
  {
    title: "5. Arama Algoritmaları",
    description: "Veri yapılarında etkili arama yöntemlerini keşfedin.",
    icon: <Search className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/arama-algoritmalari",
    topics: [
      "Linear Search",
      "Binary Search",
      "Depth-First Search (DFS)",
      "Breadth-First Search (BFS)",
      "A* Algoritması"
    ]
  },
  {
    title: "6. Dinamik Programlama",
    description: "Karmaşık problemleri alt problemlere bölerek çözmeyi öğrenin.",
    icon: <Network className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/dinamik-programlama",
    topics: [
      "Memoization",
      "Tabulation",
      "Fibonacci Serisi",
      "Knapsack Problemi",
      "Longest Common Subsequence"
    ]
  }
];

export default function DataStructuresAndAlgorithmsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Image */}
        <div className="my-8 flex justify-center">
          <Image
            src="/images/python_veri_yapilari.jpg"
            alt="Veri Yapıları ve Algoritmalar"
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
                <CardTitle>Alıştırmalar</CardTitle>
                <CardDescription>Pratik yaparak öğrenin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>LeetCode Problemleri</li>
                  <li>HackerRank Soruları</li>
                  <li>Proje Örnekleri</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Görselleştirmeler</CardTitle>
                <CardDescription>Algoritmaları görsel olarak anlayın</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Algoritma Animasyonları</li>
                  <li>Veri Yapısı Şemaları</li>
                  <li>Adım Adım Çözümler</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Kaynaklar</CardTitle>
                <CardDescription>Derinlemesine öğrenin</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Python Dokümantasyonu</li>
                  <li>Akademik Makaleler</li>
                  <li>Video Dersler</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
} 