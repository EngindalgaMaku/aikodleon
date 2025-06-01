import fs from 'fs';
import path from 'path';
import { Metadata } from 'next';
import dynamic from 'next/dynamic';

// Client Component'i dinamik olarak import ediyoruz
const MarkdownContent = dynamic(() => import('@/components/MarkdownContent'), {
  ssr: false
});

async function getPageContent(pagePath: string): Promise<string | null> {
  let filePath: string = '';
  try {
    // Projenin kök dizininden başlayarak doğru yolu belirtiyoruz.
    filePath = path.join(process.cwd(), ...pagePath.split('/'));
    console.log('Attempting to read file for page from:', filePath);
    const fileContents = fs.readFileSync(filePath, 'utf8');
    return fileContents;
  } catch (error) {
    console.error(`Markdown file for path ${pagePath} (${filePath || 'unknown path'}) not found or could not be read:`, error);
    return null;
  }
}

export async function generateMetadata(): Promise<Metadata> {
  const title = "Makine Çevirisi";
  return {
    title: `${title} | NLP | Kodleon`,
    description: `Makine Çevirisi konusu hakkında detaylı bilgiler. Kodleon yapay zeka öğrenme platformu.`,
  };
}

export default async function MachineTranslationPage() {
  // Okunacak dosyanın projenin kök dizinine göre yolu
  const markdownFilePath = 'topics/nlp/machine-translation/index.md';
  const markdownContent = await getPageContent(markdownFilePath);

  const title = "Makine Çevirisi";

  if (!markdownContent) {
    return (
      <div className="container mx-auto py-10 px-4">
        <h1 className="text-3xl font-bold mb-4">İçerik Bulunamadı</h1>
        <p>Aradığınız konu ({title}) için içerik mevcut değil veya yüklenirken bir sorun oluştu.</p>
        <p>Kontrol edilen dosya yolu: {path.join(process.cwd(), markdownFilePath)}</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen pb-16">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
          <div className="p-6 sm:p-10">
            <article className="prose lg:prose-xl dark:prose-invert max-w-none">
              {/* MarkdownContent bileşeninin props'larının doğru olduğundan emin olun */}
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