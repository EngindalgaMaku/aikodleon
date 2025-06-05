"use client";

import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { content } from './content';
import { content as mediaProcessingContent } from './medya-isleme/content';
import { content as validationContent } from './veri-dogrulama/content';
import { content as eventHandlingContent } from './olay-isleme/content';

const components = {
  h1: ({ children, ...props }: any) => (
    <h1 className="text-4xl font-extrabold mt-8 mb-4 text-primary" {...props}>{children}</h1>
  ),
  h2: ({ children, ...props }: any) => (
    <h2 className="text-2xl font-bold mt-8 mb-3 text-primary" {...props}>{children}</h2>
  ),
  h3: ({ children, ...props }: any) => (
    <h3 className="text-xl font-semibold mt-6 mb-2 text-primary" {...props}>{children}</h3>
  ),
  code: ({ className, children, ...rest }: any) => {
    const language = className?.replace("language-", "");
    return (
      <pre className="bg-[#1e1e1e] text-white p-4 rounded-lg overflow-x-auto my-6">
        <code className={className} {...rest}>
          {children}
        </code>
      </pre>
    );
  },
  div: ({ className, children, ...rest }: any) => {
    if (className === "info") {
      return (
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 mt-6 rounded-lg text-blue-900 font-medium shadow-sm" {...rest}>
          {children}
        </div>
      );
    }
    if (className === "warning") {
      return (
        <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 mb-6 mt-6 rounded-lg text-yellow-900 font-medium shadow-sm" {...rest}>
          {children}
        </div>
      );
    }
    if (className === "tip") {
      return (
        <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-6 mt-6 rounded-lg text-green-900 font-medium shadow-sm" {...rest}>
          {children}
        </div>
      );
    }
    return <div className={className} {...rest}>{children}</div>;
  },
  a: ({ href, children, ...rest }: any) => {
    // Alıştırma linklerini yönlendir
    if (href?.includes("/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler/")) {
      const exercise = href.split("/").pop();
      let content;
      switch (exercise) {
        case "medya-isleme":
          content = mediaProcessingContent;
          break;
        case "veri-dogrulama":
          content = validationContent;
          break;
        case "olay-isleme":
          content = eventHandlingContent;
          break;
      }
      if (content) {
        return (
          <button 
            onClick={() => {
              const exerciseDiv = document.getElementById(exercise);
              if (exerciseDiv) {
                exerciseDiv.scrollIntoView({ behavior: "smooth" });
              }
            }}
            className="text-blue-600 hover:text-blue-800 underline"
            {...rest}
          >
            {children}
          </button>
        );
      }
    }
    return <a href={href} className="text-blue-600 hover:text-blue-800 underline" {...rest}>{children}</a>;
  }
};

export default function Page() {
  return (
    <div className="max-w-4xl mx-auto p-4 space-y-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/nesneye-yonelik-programlama">
            <ArrowLeft className="h-4 w-4" />
            OOP Konularına Dön
          </Link>
        </Button>
      </div>

      <div className="prose dark:prose-invert max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {content}
        </ReactMarkdown>
      </div>

      <div id="medya-isleme" className="prose dark:prose-invert max-w-none mt-8 pt-8 border-t">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {mediaProcessingContent}
        </ReactMarkdown>
      </div>

      <div id="veri-dogrulama" className="prose dark:prose-invert max-w-none mt-8 pt-8 border-t">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {validationContent}
        </ReactMarkdown>
      </div>

      <div id="olay-isleme" className="prose dark:prose-invert max-w-none mt-8 pt-8 border-t">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {eventHandlingContent}
        </ReactMarkdown>
      </div>

      {/* Navigasyon Linkleri */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/cok-bicimlilk">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Çok Biçimlilik
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/tasarim-desenleri">
            Sonraki Konu: Tasarım Desenleri
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 