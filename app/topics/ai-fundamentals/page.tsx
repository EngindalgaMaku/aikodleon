import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

async function getPageData() {
  try {
    const filePath = path.join(process.cwd(), 'topics/ai-fundamentals.md');
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const { data, content } = matter(fileContent);
    return { metadata: data, content };
  } catch (error) {
    console.error(`Error reading or parsing markdown file:`, error);
    return null;
  }
}

export async function generateMetadata(): Promise<Metadata> {
  const data = await getPageData();
  
  if (!data) {
    return {
      title: 'Yapay Zeka Temelleri | Kodleon',
      description: 'Yapay zeka alanının temel kavramları, teknikleri ve uygulama alanlarını keşfedin.',
    };
  }
  
  const { metadata } = data;
  return {
    title: `${metadata.title} | Kodleon`,
    description: metadata.description,
    openGraph: {
      title: `${metadata.title} | Kodleon`,
      description: metadata.description,
      images: [{ url: metadata.image || 'https://kodleon.com/blog-images/ai-fundamentals.jpg' }],
    }
  };
}

export default async function AIFundamentalsPage() {
  const data = await getPageData();

  if (!data) {
    return (
      <div className="container mx-auto py-10 px-4">
        <h1 className="text-3xl font-bold">İçerik Yüklenemedi</h1>
        <p>Yapay Zeka Temelleri konusu için içerik bulunamadı veya yüklenirken bir sorun oluştu.</p>
        <Button asChild variant="ghost" className="mt-4">
          <Link href="/topics">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Konulara Geri Dön
          </Link>
        </Button>
      </div>
    );
  }
  
  const { metadata, content } = data;

  return (
    <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen pb-16">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="mb-8">
          <Button asChild variant="outline" size="sm" className="gap-1 text-sm">
            <Link href="/topics">
              <ArrowLeft className="h-4 w-4" />
              Geri: Tüm Konular
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
        <div className="mt-16 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 