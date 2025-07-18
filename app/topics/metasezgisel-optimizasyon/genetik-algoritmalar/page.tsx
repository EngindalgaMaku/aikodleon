import fs from 'fs';
import path from 'path';
import { Metadata } from 'next';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import matter from 'gray-matter';

const MarkdownContent = dynamic(() => import('@/components/MarkdownContent'), {
  ssr: false
});

// This function reads and parses the Markdown file
async function getPageContent(pagePath: string) {
  try {
    const filePath = path.join(process.cwd(), pagePath);
    const fileContents = fs.readFileSync(filePath, 'utf8');
    const { data, content } = matter(fileContents);
    return { metadata: data, content };
  } catch (error) {
    console.error(`Error processing markdown file for path ${pagePath}:`, error);
    return null;
  }
}

// Generate metadata for the page
export async function generateMetadata(): Promise<Metadata> {
  const pageData = await getPageContent('topics/metasezgisel-optimizasyon/genetik-algoritmalar.md');
  
  if (!pageData) {
    return {
      title: "İçerik Bulunamadı",
      description: "Aranan içerik mevcut değil."
    };
  }
  
  const { metadata } = pageData;
  const pageTitle = metadata.title || "Genetik Algoritmalar";
  const description = metadata.description || "Genetik Algoritmaların (GA) temellerini, çalışma prensiplerini ve uygulama alanlarını öğrenin.";

  return {
    title: `${pageTitle} | Metasezgisel Optimizasyon | Kodleon`,
    description: description,
    keywords: metadata.keywords || "genetik algoritmalar, ga, metasezgisel optimizasyon, evrimsel hesaplama, optimizasyon, yapay zeka",
     openGraph: {
      title: `${pageTitle} | Kodleon`,
      description: description,
      images: [{ url: metadata.image || 'https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2' }],
    }
  };
}

// Page component
export default async function GeneticAlgorithmsPage() {
  const pageData = await getPageContent('topics/metasezgisel-optimizasyon/genetik-algoritmalar.md');
  
  if (!pageData) {
    const markdownFilePath = 'topics/metasezgisel-optimizasyon/genetik-algoritmalar.md';
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
        <p>Aradığınız konu için içerik mevcut değil veya yüklenirken bir sorun oluştu.</p>
        <p>Kontrol edilen dosya yolu: {path.join(process.cwd(), markdownFilePath)}</p>
      </div>
    );
  }

  const { content } = pageData;

  return (
    <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen pb-16">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="mb-8">
          <Button asChild variant="outline" size="sm" className="gap-1 text-sm">
            <Link href="/topics/metasezgisel-optimizasyon">
              <ArrowLeft className="h-4 w-4" />
              Geri: Metasezgisel Optimizasyon
            </Link>
          </Button>
        </div>
        <div className="bg-white dark:bg-gray-850 rounded-xl shadow-xl overflow-hidden">
          <div className="p-6 sm:p-10">
            <article className="prose prose-lg lg:prose-xl dark:prose-invert max-w-none">
              <MarkdownContent content={content} />
            </article>
          </div>
        </div>
        <div className="mt-12 text-center">
          <Button asChild variant="default">
            <Link href="/topics/metasezgisel-optimizasyon/parcacik-suru-optimizasyonu">
              Sonraki Konu: Parçacık Sürü Optimizasyonu <ArrowRight className="ml-2 h-4 w-4" />
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
