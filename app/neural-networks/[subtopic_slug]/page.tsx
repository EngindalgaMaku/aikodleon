import fs from 'fs';
import path from 'path';
import { Metadata } from 'next';
import dynamic from 'next/dynamic';

// Client Component'i dinamik olarak import ediyoruz
const MarkdownContent = dynamic(() => import('@/components/MarkdownContent'), {
  ssr: false
});

// Markdown içeriğini işlemek için (henüz bir kütüphane seçilmedi)
// import ReactMarkdown from 'react-markdown'; 

interface SubtopicPageProps {
  params: {
    subtopic_slug: string;
  };
}

async function getMarkdownContent(slug: string): Promise<string | null> {
  try {
    const filePath = path.join(process.cwd(), `${slug}.md`);
    const fileContents = fs.readFileSync(filePath, 'utf8');
    return fileContents;
  } catch (error) {
    console.error(`Markdown file for slug ${slug} not found or could not be read:`, error);
    return null;
  }
}

export async function generateMetadata({ params }: SubtopicPageProps): Promise<Metadata> {
  const { subtopic_slug } = params;
  // Slug'dan başlık oluştur (örneğin: temel-sinir-agi-mimarileri -> Temel Sinir Ağı Mimarileri)
  const title = subtopic_slug
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return {
    title: `${title} | Sinir Ağları | Kodleon`,
    description: `${title} konusu hakkında detaylı bilgiler. Kodleon yapay zeka öğrenme platformu.`,
  };
}

export default async function SubtopicPage({ params }: SubtopicPageProps) {
  const { subtopic_slug } = params;
  const markdownContent = await getMarkdownContent(subtopic_slug);

  const title = subtopic_slug
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  if (!markdownContent) {
    return (
      <div className="container mx-auto py-10">
        <h1 className="text-3xl font-bold mb-4">İçerik Bulunamadı</h1>
        <p>Aradığınız konu ({subtopic_slug}) için içerik mevcut değil veya yüklenirken bir sorun oluştu.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-10">
      <article className="prose lg:prose-xl dark:prose-invert max-w-none">
        <MarkdownContent content={markdownContent} />
      </article>
    </div>
  );
}

// Dinamik yolları oluşturmak için
export async function generateStaticParams() {
  // Markdown dosyalarının listesi
  const subtopicFiles = [
    'temel-sinir-agi-mimarileri.md',
    'yapay-sinir-aglari-guvenlik-uygulamalari.md',
    'tekrarlayan-sinir-aglari-rnn.md',
    'transformerlar.md',
    'konvolusyonel-sinir-aglari.md'
  ];
  
  return subtopicFiles.map(file => ({ 
    subtopic_slug: file.replace('.md', '') 
  }));
} 