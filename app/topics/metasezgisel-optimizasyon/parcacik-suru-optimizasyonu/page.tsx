import fs from 'fs';
import path from 'path';
import { Metadata } from 'next';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const MarkdownContent = dynamic(() => import('@/components/MarkdownContent'), {
  ssr: false
});

async function getPageContent(pagePath: string): Promise<string | null> {
  let filePath: string = '';
  try {
    filePath = path.join(process.cwd(), ...pagePath.split('/'));
    const fileContents = fs.readFileSync(filePath, 'utf8');
    return fileContents;
  } catch (error) {
    console.error(`Markdown file for path ${pagePath} (${filePath || 'unknown path'}) not found or could not be read:`, error);
    return null;
  }
}

export async function generateMetadata(): Promise<Metadata> {
  const title = "Parçacık Sürü Optimizasyonu (PSO)";
  const description = "Parçacık Sürü Optimizasyonu (PSO) algoritmasının temellerini, çalışma prensiplerini ve uygulama alanlarını keşfedin.";
  return {
    title: `${title} | Metasezgisel Optimizasyon | Kodleon`,
    description: description,
    keywords: "parçacık sürü optimizasyonu, pso, metasezgisel optimizasyon, sürü zekası, optimizasyon, yapay zeka",
    openGraph: {
      title: `${title} | Kodleon`,
      description: description,
      images: [{ url: 'https://images.pexels.com/photos/1089842/pexels-photo-1089842.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2' }],
    }
  };
}

export default async function ParticleSwarmOptimizationPage() {
  const markdownFilePath = 'topics/metasezgisel-optimizasyon/parcacik-suru-optimizasyonu.md';
  const markdownContent = await getPageContent(markdownFilePath);
  const pageTitle = "Parçacık Sürü Optimizasyonu";

  if (!markdownContent) {
    return (
      <div className="container mx-auto py-10 px-4">
        <div className="mb-6">
          <Button asChild variant="ghost" size="sm" className="gap-1">
            <Link href="/topics/metasezgisel-optimizasyon">
              <ArrowLeft className="h-4 w-4" />
              Metasezgisel Optimizasyon
            </Link>
          </Button>
        </div>
        <h1 className="text-3xl font-bold mb-4">İçerik Bulunamadı</h1>
        <p>Aradığınız konu ({pageTitle}) için içerik mevcut değil veya yüklenirken bir sorun oluştu.</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen pb-16">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="mb-8 flex justify-between items-center">
          <Button asChild variant="outline" size="sm" className="gap-1 text-sm">
            <Link href="/topics/metasezgisel-optimizasyon/genetik-algoritmalar">
              <ArrowLeft className="h-4 w-4" />
              Önceki: Genetik Algoritmalar
            </Link>
          </Button>
          <Button asChild variant="outline" size="sm" className="gap-1 text-sm">
            <Link href="/topics/metasezgisel-optimizasyon">
              <ArrowLeft className="h-4 w-4" /> 
              Ana Kategori: Metasezgisel Optimizasyon
            </Link>
          </Button>
        </div>
        <div className="bg-white dark:bg-gray-850 rounded-xl shadow-xl overflow-hidden">
          <div className="p-6 sm:p-10">
            <article className="prose prose-lg lg:prose-xl dark:prose-invert max-w-none">
              <MarkdownContent content={markdownContent} />
            </article>
          </div>
        </div>
        <div className="mt-12 text-center">
          <Button asChild variant="default">
            <Link href="/topics/metasezgisel-optimizasyon/tavlama-benzetimi">
              Sonraki Konu: Tavlama Benzetimi <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
        <div className="mt-16 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 