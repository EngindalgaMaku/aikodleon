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
  let filePath: string = ''; // filePath'i try bloğunun dışında tanımla
  try {
    filePath = path.join(process.cwd(), 'neural-networks', `${slug}.md`);
    console.log('Attempting to read file from:', filePath);
    const fileContents = fs.readFileSync(filePath, 'utf8');
    return fileContents;
  } catch (error) {
    // Hata mesajında belirtilen formatı kullanarak filePath'i de logla
    console.error(`Markdown file for slug ${slug} not found or could not be read at ${filePath || 'unknown path'}:`, error);
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
      <div className="container mx-auto py-10 px-4">
        <h1 className="text-3xl font-bold mb-4">İçerik Bulunamadı</h1>
        <p>Aradığınız konu ({subtopic_slug}) için içerik mevcut değil veya yüklenirken bir sorun oluştu.</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen pb-16">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
          <div className="p-6 sm:p-10">
            <article className="prose lg:prose-xl dark:prose-invert max-w-none">
              <MarkdownContent content={markdownContent} />
            </article>
          </div>
        </div>
        
        <div className="mt-10 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
          <p className="mt-2">Bu içerik sadece eğitim amaçlıdır ve sürekli güncellenmektedir.</p>
        </div>
      </div>
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