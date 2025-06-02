import fs from 'fs';
import path from 'path';
import { Metadata } from 'next';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft, ArrowRight, Home } from 'lucide-react';

import { Button } from '@/components/ui/button';

const MarkdownContent = dynamic(() => import('@/components/MarkdownContent'), {
  ssr: false,
});

const contentDir = path.join(process.cwd(), 'topics/metasezgisel-optimizasyon');

async function getPageContent(pagePath: string) {
  try {
    const markdown = fs.readFileSync(pagePath, 'utf-8');
    return markdown;
  } catch (error) {
    console.error('Error reading markdown file:', error);
    return null;
  }
}

export async function generateMetadata(): Promise<Metadata> {
  const title = "Diferansiyel Gelişim (DE) | Metasezgisel Optimizasyon";
  const description = "Diferansiyel Gelişim (DE) algoritmasının temellerini, adımlarını, avantajlarını, dezavantajlarını ve uygulama alanlarını öğrenin.";
  const keywords = ["Diferansiyel Gelişim", "Differential Evolution", "DE", "Metasezgisel Optimizasyon", "Evrimsel Algoritmalar", "Sürekli Optimizasyon", "Optimizasyon Teknikleri"];
  const locale = "tr_TR";
  const type = "article";
  const url = "/topics/metasezgisel-optimizasyon/diferansiyel-gelisim";
  //const imageUrl = "/images/topics/differential-evolution.jpg"; // Güncellenecek

  return {
    title,
    description,
    keywords,
    openGraph: {
      title,
      description,
      type,
      url,
      locale,
      //images: imageUrl,
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      //images: [imageUrl],
    },
  };
}

export default async function DifferentialEvolutionPage() {
  const pagePath = path.join(contentDir, 'diferansiyel-gelisim.md');
  const content = await getPageContent(pagePath);

  if (!content) {
    return <div className="container mx-auto px-4 py-8">İçerik yüklenemedi. Lütfen daha sonra tekrar deneyin.</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <MarkdownContent content={content} />
      <div className="mt-8 flex flex-col sm:flex-row justify-between space-y-4 sm:space-y-0 sm:space-x-4">
        <Button asChild variant="outline">
          <Link href="/topics/metasezgisel-optimizasyon/yapay-ari-kolonisi-algoritmasi" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Önceki Konu: Yapay Arı Kolonisi Algoritması
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/topics/metasezgisel-optimizasyon" className="flex items-center">
            <Home className="mr-2 h-4 w-4" />
            Ana Kategori: Metasezgisel Optimizasyon
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/topics/metasezgisel-optimizasyon/uyum-aramasi" className="flex items-center">
            Sonraki Konu: Uyum Araması <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  );
} 