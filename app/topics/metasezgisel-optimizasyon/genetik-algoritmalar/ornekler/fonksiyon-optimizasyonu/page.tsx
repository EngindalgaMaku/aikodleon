import { getPostBySlug } from '@/lib/markdown';
import MarkdownContent from '@/components/MarkdownContent';
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { siteConfig } from '@/config/site'; // Site yapılandırmasını import ediyoruz

interface PageProps {
  params: {
    // Bu sayfanın 'slug' parametresi olmayacak çünkü dosya adı sabit.
    // Ancak, gelecekteki dinamik örnekler için bu yapıyı koruyabiliriz
    // veya daha spesifik bir yol izleyebiliriz. Şimdilik,
    // getPostBySlug fonksiyonunu doğrudan dosya adı ile çağıracağız.
  };
}

const MARKDOWN_FILE_PATH = "metasezgisel-optimizasyon/genetik-algoritmalar/ornekler/fonksiyon-optimizasyonu.md";

export async function generateMetadata(): Promise<Metadata> {
  const post = await getPostBySlug(MARKDOWN_FILE_PATH);

  if (!post) {
    return {};
  }

  const ogImageUrl = `${siteConfig.url}/api/og?title=${encodeURIComponent(post.frontmatter.title || 'Genetik Algoritma Örneği')}`;

  return {
    title: post.frontmatter.title,
    description: post.frontmatter.description,
    keywords: post.frontmatter.keywords?.split(',').map((k: string) => k.trim()),
    openGraph: {
      type: 'article',
      url: `${siteConfig.url}/topics/metasezgisel-optimizasyon/genetik-algoritmalar/ornekler/fonksiyon-optimizasyonu`,
      title: post.frontmatter.title,
      description: post.frontmatter.description,
      images: [{ url: ogImageUrl }],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.frontmatter.title,
      description: post.frontmatter.description,
      images: [ogImageUrl],
    },
  };
}

export default async function GeneticAlgorithmFonksiyonOptimizasyonuPage({ params }: PageProps) {
  const post = await getPostBySlug(MARKDOWN_FILE_PATH);

  if (!post) {
    notFound();
  }

  console.log("Markdown Content Slug:", MARKDOWN_FILE_PATH);
  console.log("Extracted post.content:", post.rawContent); // post.content içeriğini konsola yazdır

  return (
    <article className="container mx-auto py-8 max-w-3xl">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
          {post.frontmatter.title}
        </h1>
        {post.frontmatter.description && (
          <p className="mt-2 text-lg text-muted-foreground">
            {post.frontmatter.description}
          </p>
        )}
      </header>
      <MarkdownContent content={post.rawContent} />
      {/* İleride buraya "Sonraki Örnek" veya "Örnek Listesine Dön" gibi linkler eklenebilir */}
    </article>
  );
} 