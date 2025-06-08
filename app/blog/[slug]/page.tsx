import { notFound } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { getPostBySlug, getAllPosts } from '@/lib/mdx';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';

// Gelişmiş Markdown işleyici oluştur
const md: MarkdownIt = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return '<pre class="hljs"><code>' +
               hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
               '</code></pre>';
      } catch (__) {}
    }
    return '<pre class="hljs"><code>' + md.utils.escapeHtml(str) + '</code></pre>';
  }
});

// Markdown'da satır başlarını koruyan eklenti
md.use(require('markdown-it-attrs'));

interface BlogPostPageProps {
  params: {
    slug: string;
  };
}

export async function generateStaticParams() {
  const posts = await getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

// Markdown içeriğini hazırlayan yardımcı fonksiyon
function processContent(content: string) {
  // Başlıkları düzgün işle (# ile başlayan satırlar)
  content = content.replace(/^(#{1,6})\s+(.+)$/gm, (match, hashes, title) => {
    const level = hashes.length;
    return `${hashes} ${title}`;
  });

  // Liste öğelerini düzgün işle (- veya * ile başlayan satırlar)
  content = content.replace(/^(\s*)[-*]\s+(.+)$/gm, (match, indent, text) => {
    return `${indent}- ${text}`;
  });

  // Kod bloklarını koruyarak içeriği işle
  content = content.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `\`\`\`${lang}\n${code}\`\`\``;
  });

  // İnline kod bloklarını koru
  content = content.replace(/`([^`]+)`/g, (match, code) => {
    return `\`${code}\``;
  });

  // Bağlantıları düzgün işle
  content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
    return `[${text}](${url})`;
  });

  return content;
}

export default async function BlogPostPage({ params }: BlogPostPageProps) {
  const post = await getPostBySlug(params.slug).catch(() => null);

  if (!post) {
    notFound();
  }

  // Markdown içeriğini HTML'e dönüştür
  const htmlContent = post.content ? md.render(processContent(post.content)) : '';

  return (
    <div className="container max-w-4xl mx-auto py-12">
      <div className="mb-6">
        <Button variant="ghost" asChild className="mb-6">
          <Link href="/blog" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Blog
          </Link>
        </Button>
      </div>

      <article className="prose prose-lg dark:prose-invert mx-auto">
        {post.image && (
          <div className="relative w-full h-[400px] mb-8 rounded-lg overflow-hidden">
            <Image
              src={post.image}
              alt={post.title}
              fill
              className="object-cover"
            />
          </div>
        )}

        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-4">{post.title}</h1>
          <div className="flex items-center gap-4 text-muted-foreground">
            <span>{post.author}</span>
            <span>•</span>
            <time dateTime={post.date}>
              {new Date(post.date).toLocaleDateString('tr-TR', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              })}
            </time>
          </div>
        </div>

        {post.content && (
          <div 
            className="markdown-content"
            dangerouslySetInnerHTML={{ __html: htmlContent }} 
          />
        )}

        {post.tags && post.tags.length > 0 && (
          <div className="mt-8 pt-8 border-t">
            <div className="flex flex-wrap gap-2">
              {post.tags.map((tag: string, index: number) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-muted rounded-full text-sm"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </article>
    </div>
  );
} 