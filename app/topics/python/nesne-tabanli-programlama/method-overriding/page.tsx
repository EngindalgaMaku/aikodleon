import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

const contentDir = path.join(process.cwd(), 'topics/python/nesne-tabanli-programlama');

async function getPageData(page: string) {
  try {
    const filePath = path.join(contentDir, `${page}.md`);
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const { data, content } = matter(fileContent);
    return { metadata: data, content };
  } catch (error) {
    console.error(`Error reading or parsing markdown file for ${page}:`, error);
    return null;
  }
}

export async function generateMetadata(): Promise<Metadata> {
  const data = await getPageData('method-overriding');
  
  if (!data) {
    return {
      title: 'İçerik Bulunamadı',
      description: 'Aradığınız içerik mevcut değil.',
    };
  }
  
  const { metadata } = data;
  return {
    title: `${metadata.title} | Python OOP | Kodleon`,
    description: metadata.description,
  };
}

export default async function MethodOverridingPage() {
  const data = await getPageData('method-overriding');

  if (!data) {
    return (
      <div className="container mx-auto py-10 px-4">
        <h1 className="text-3xl font-bold">İçerik Yüklenemedi</h1>
        <p>Method Overriding konusu için içerik bulunamadı veya yüklenirken bir sorun oluştu.</p>
        <Button asChild variant="ghost" className="mt-4">
          <Link href="/topics/python/nesne-tabanli-programlama/cok-bicimlilik">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Geri Dön
          </Link>
        </Button>
      </div>
    );
  }
  
  const { metadata, content } = data;

  return (
    <div className="container mx-auto py-10 px-4 max-w-4xl">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm">
          <Link href="/topics/python/nesne-tabanli-programlama/cok-bicimlilik">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Geri: Çok Biçimlilik
          </Link>
        </Button>
      </div>
      <article className="prose prose-lg lg:prose-xl dark:prose-invert max-w-none">
        <h1>{metadata.title}</h1>
        <p className="lead">{metadata.description}</p>
        <MarkdownContent content={content} />
      </article>
      <div className="mt-12 flex justify-between">
         <Button asChild variant="outline">
            <Link href="/topics/python/nesne-tabanli-programlama/cok-bicimlilik">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Önceki Konu: Çok Biçimlilik
            </Link>
        </Button>
        <Button asChild>
            <Link href="/topics/python/nesne-tabanli-programlama/soyut-siniflar">
                Sonraki Konu: Soyut Sınıflar ve Arayüzler
                <ArrowRight className="h-4 w-4 ml-2" />
            </Link>
        </Button>
      </div>
    </div>
  );
} 