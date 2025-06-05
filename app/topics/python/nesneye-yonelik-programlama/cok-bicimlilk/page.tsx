"use client";

import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";
import { content } from "./content";
import { content as mediaPlayerContent } from "./medya-oynatici/content";
import { content as shapeDrawingContent } from "./sekil-cizim/content";
import { content as gameCharacterContent } from "./oyun-karakter/content";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const components = {
  code: ({ className, children, ...rest }: any) => {
    const language = className?.replace("language-", "");
    return (
      <pre className="bg-[#1e1e1e] text-white p-4 rounded-lg overflow-x-auto">
        <code className={className} {...rest}>
          {children}
        </code>
      </pre>
    );
  },
  div: ({ className, children, ...rest }: any) => {
    if (className === "info") {
      return (
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4 rounded-lg" {...rest}>
          {children}
        </div>
      );
    }
    if (className === "warning") {
      return (
        <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 mb-4 rounded-lg" {...rest}>
          {children}
        </div>
      );
    }
    if (className === "tip") {
      return (
        <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-4 rounded-lg" {...rest}>
          {children}
        </div>
      );
    }
    return <div className={className} {...rest}>{children}</div>;
  },
  a: ({ href, children, ...rest }: any) => {
    // Alıştırma linklerini yönlendir
    if (href?.includes("/topics/python/nesneye-yonelik-programlama/cok-bicimlilk/")) {
      const exercise = href.split("/").pop();
      let content;
      switch (exercise) {
        case "medya-oynatici":
          content = mediaPlayerContent;
          break;
        case "sekil-cizim":
          content = shapeDrawingContent;
          break;
        case "oyun-karakter":
          content = gameCharacterContent;
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

      <div id="medya-oynatici" className="prose dark:prose-invert max-w-none mt-8 pt-8 border-t">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {mediaPlayerContent}
        </ReactMarkdown>
      </div>

      <div id="sekil-cizim" className="prose dark:prose-invert max-w-none mt-8 pt-8 border-t">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {shapeDrawingContent}
        </ReactMarkdown>
      </div>

      <div id="oyun-karakter" className="prose dark:prose-invert max-w-none mt-8 pt-8 border-t">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {gameCharacterContent}
        </ReactMarkdown>
      </div>

      {/* Navigasyon Linkleri */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/kapsulleme">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Kapsülleme
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler">
            Sonraki Konu: Soyut Sınıflar ve Arayüzler
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