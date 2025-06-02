import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getPostBySlug } from '@/lib/markdown';
import { Metadata } from 'next';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Genetik Algoritma Örnekleri (Python) | Kodleon Metasezgisel Optimizasyon',
  description: 'Python ile Genetik Algoritma örnekleri ve pratik uygulamaları. Fonksiyon optimizasyonu, Gezgin Satıcı Problemi (TSP) ve Sırt Çantası Problemi gibi örneklere göz atın.',
  keywords: 'genetik algoritma örnekleri, python genetik algoritma, tsp python, sırt çantası problemi python, metasezgisel optimizasyon örnekleri, kodleon',
  alternates: {
    canonical: '/topics/metasezgisel-optimizasyon/genetik-algoritmalar/genetik-algoritma-ornekleri',
  },
  openGraph: {
    title: 'Genetik Algoritma Örnekleri (Python) | Kodleon',
    description: 'Python ile Genetik Algoritma örnekleri ve pratik uygulamalarını keşfedin.',
    url: '/topics/metasezgisel-optimizasyon/genetik-algoritmalar/genetik-algoritma-ornekleri',
    images: [
      {
        url: '/images/placeholder-genetik-algoritma.png', // Genel bir placeholder görsel, daha sonra güncellenebilir
        width: 1200,
        height: 630,
        alt: 'Genetik Algoritma Örnekleri - Kodleon',
      },
    ],
  },
};

export default async function GeneticAlgorithmExamplesPage() {
  const { rawContent, frontmatter } = getPostBySlug('metasezgisel-optimizasyon/genetik-algoritma-ornekleri');

  return (
    <div className="container max-w-4xl mx-auto py-12 px-4">
      <Button asChild variant="ghost" size="sm" className="mb-6">
        <Link href="/topics/metasezgisel-optimizasyon/genetik-algoritmalar">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Genetik Algoritmalar konusuna geri dön
        </Link>
      </Button>

      <article className="max-w-none">
        {frontmatter.title && <h1 className="text-4xl font-bold mt-8 mb-6 text-gray-800 dark:text-gray-100">{frontmatter.title}</h1>}
        <MarkdownContent content={rawContent} />
      </article>
    </div>
  );
} 