import { Metadata } from 'next';
import Link from 'next/link';
import { 
    ArrowLeft, 
    List, 
    Binary, 
    GitFork, 
    Box, 
    Search, 
    Network 
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Veri Yapıları ve Algoritmalar',
  description: 'Python programlama dilinde temel ve ileri düzey veri yapıları ile algoritmaları öğrenin.',
};

const introContent = `
# Python ile Veri Yapıları ve Algoritmalar
Veri yapıları ve algoritmalar, etkili ve verimli programlar yazmanın temelidir. Bu bölümde, Python kullanarak temelden ileri seviyeye veri yapılarını ve algoritmaları öğrenecek, bunları gerçek dünya problemlerinde nasıl uygulayacağınızı keşfedeceksiniz.
`;

const cardColors = [
  'bg-gradient-to-br from-sky-50 to-sky-100 dark:from-sky-900/20 dark:to-sky-800/20 border-sky-200 dark:border-sky-800',
  'bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-900/20 dark:to-emerald-800/20 border-emerald-200 dark:border-emerald-800',
  'bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-900/20 dark:to-indigo-800/20 border-indigo-200 dark:border-indigo-800',
  'bg-gradient-to-br from-fuchsia-50 to-fuchsia-100 dark:from-fuchsia-900/20 dark:to-fuchsia-800/20 border-fuchsia-200 dark:border-fuchsia-800',
  'bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 border-amber-200 dark:border-amber-800',
  'bg-gradient-to-br from-rose-50 to-rose-100 dark:from-rose-900/20 dark:to-rose-800/20 border-rose-200 dark:border-rose-800',
];

const sections = [
  {
    title: "1. Temel Veri Yapıları",
    description: "Python'da yerleşik ve temel veri yapılarını öğrenin.",
    icon: <List className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/temel-veri-yapilari",
    topics: ["Listeler ve Tuple'lar", "Dictionary ve Set'ler", "Stack ve Queue", "Linked List"]
  },
  {
    title: "2. İleri Veri Yapıları",
    description: "Karmaşık veri yapılarını ve uygulamalarını keşfedin.",
    icon: <Binary className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/ileri-veri-yapilari",
    topics: ["Ağaçlar (Trees)", "Heap ve Priority Queue", "Hash Table", "Graf (Graph)"]
  },
  {
    title: "3. Temel Algoritmalar",
    description: "Algoritmaların temellerini ve analiz yöntemlerini öğrenin.",
    icon: <GitFork className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/temel-algoritmalar",
    topics: ["Algoritma Analizi (Big O)", "Recursion", "Divide and Conquer", "Greedy Algoritmalar"]
  },
  {
    title: "4. Sıralama Algoritmaları",
    description: "Farklı sıralama algoritmalarını ve senaryolarını öğrenin.",
    icon: <Box className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/siralama-algoritmalari",
    topics: ["Bubble Sort", "Insertion Sort", "Merge Sort", "Quick Sort"]
  },
  {
    title: "5. Arama Algoritmaları",
    description: "Veri yapılarında etkili arama yöntemlerini keşfedin.",
    icon: <Search className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/arama-algoritmalari",
    topics: ["Linear Search", "Binary Search", "DFS", "BFS"]
  },
  {
    title: "6. Dinamik Programlama",
    description: "Karmaşık problemleri alt problemlere bölerek çözün.",
    icon: <Network className="h-6 w-6" />,
    href: "/topics/python/veri-yapilari-ve-algoritmalar/dinamik-programlama",
    topics: ["Memoization", "Tabulation", "Knapsack Problemi", "LCS"]
  }
];

export default function DataStructuresAndAlgorithmsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild>
            <Link href="/topics/python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python Eğitimleri
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert mb-12">
          <MarkdownContent content={introContent} />
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