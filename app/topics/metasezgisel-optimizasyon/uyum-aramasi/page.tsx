import fs from 'fs';
import path from 'path';
import { Metadata } from 'next';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft, Home } from 'lucide-react'; // ArrowRight kaldırıldı

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
  const title = "Uyum Araması (HS) | Metasezgisel Optimizasyon";
  const description = "Uyum Araması (Harmony Search - HS) algoritmasının temellerini, adımlarını, avantajlarını, dezavantajlarını ve uygulama alanlarını öğrenin.";
  const keywords = ["Uyum Araması", "Harmony Search", "HS", "Metasezgisel Optimizasyon", "Müzik Esinli Algoritmalar", "Optimizasyon Teknikleri"];
  const locale = "tr_TR";
  const type = "article";
  const url = "/topics/metasezgisel-optimizasyon/uyum-aramasi";
  //const imageUrl = "/images/topics/harmony-search.jpg"; // Güncellenecek

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

export default async function HarmonySearchPage() {
  const pagePath = path.join(contentDir, 'uyum-aramasi.md');
  const content = await getPageContent(pagePath);

  if (!content) {
    return <div className="container mx-auto px-4 py-8">İçerik yüklenemedi. Lütfen daha sonra tekrar deneyin.</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <MarkdownContent content={content} />
      <div className="mt-8 flex flex-col sm:flex-row justify-between space-y-4 sm:space-y-0 sm:space-x-4">
        <Button asChild variant="outline">
          <Link href="/topics/metasezgisel-optimizasyon/diferansiyel-gelisim" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Önceki Konu: Diferansiyel Gelişim
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/topics/metasezgisel-optimizasyon" className="flex items-center">
            <Home className="mr-2 h-4 w-4" />
            Ana Kategori: Metasezgisel Optimizasyon
          </Link>
        </Button>
        {/* Son konu olduğu için Sonraki Konu butonu yok */}
      </div>
    </div>
  );
} 