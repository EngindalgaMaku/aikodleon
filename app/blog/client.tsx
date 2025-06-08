'use client';

import Link from "next/link";
import Image from "next/image";
import { ArrowRight, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";

const POSTS_PER_PAGE = 6;

interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  author: string;
  category: string;
  image?: string;
}

const defaultImages: { [key: string]: string } = {
  "python-yapay-zeka-2025": "/blog-images/ai-future-2025.jpg",
  "yapay-zeka-tarim-uygulamalari": "/blog-images/ai-agriculture.jpg",
  "python-ile-endustriyel-otomasyon": "/blog-images/industrial-automation.jpg",
  "python-ve-yapay-zeka-2024-trendleri": "/blog-images/ai-trends-2024.jpg",
  "ai-video-uretimi-veo-flow": "/blog-images/ai-video.jpg",
  "embodied-ai-future": "/blog-images/embodied-ai.jpg",
  "ai-kod-asistanlari-karsilastirmasi": "/blog-images/ai-coding.jpg"
};

export default function BlogPage({ posts }: { posts: BlogPost[] }) {
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = Math.ceil(posts.length / POSTS_PER_PAGE);
  
  const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
  const endIndex = Math.min(startIndex + POSTS_PER_PAGE, posts.length);
  
  const currentPagePosts = posts.slice(startIndex, endIndex);

  return (
    <div className="container max-w-6xl mx-auto py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Blog</h1>
        <p className="text-xl text-muted-foreground">
          Yapay zeka dünyasındaki son gelişmeler, teknoloji trendleri ve uzman görüşleri.
        </p>
        <p className="text-sm text-muted-foreground mt-2">
          Toplam {posts.length} yazı
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {currentPagePosts.map((post, index) => (
          <Card key={index} className="overflow-hidden">
            <div className="relative h-48">
              <Image 
                src={post.image || defaultImages[post.slug] || "/blog-images/default.jpg"}
                alt={post.title}
                fill
                className="object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
              <div className="absolute top-4 left-4">
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-background/80 backdrop-blur-sm">
                  {post.category}
                </span>
              </div>
            </div>
            <CardHeader>
              <CardTitle>{post.title}</CardTitle>
              <CardDescription>{post.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>{post.author}</span>
                <span>{new Date(post.date).toLocaleDateString('tr-TR', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}</span>
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild variant="ghost" className="gap-1 ml-auto">
                <Link href={`/blog/${post.slug}`}>
                  Devamını Oku
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      {/* Pagination Status */}
      <div className="text-center mt-8 mb-4 text-sm text-muted-foreground">
        {startIndex + 1}-{endIndex} arası yazılar gösteriliyor (toplam {posts.length} yazı)
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2 mt-4">
          <Button
            variant="outline"
            size="sm"
            disabled={currentPage <= 1}
            onClick={() => setCurrentPage(currentPage - 1)}
            className="px-3"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Önceki
          </Button>
          
          <div className="flex items-center gap-1 mx-2">
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((pageNum) => (
              <Button
                key={pageNum}
                variant={pageNum === currentPage ? "default" : "outline"}
                size="sm"
                onClick={() => setCurrentPage(pageNum)}
                className="w-8 h-8 p-0"
              >
                {pageNum}
              </Button>
            ))}
          </div>

          <Button
            variant="outline"
            size="sm"
            disabled={currentPage >= totalPages}
            onClick={() => setCurrentPage(currentPage + 1)}
            className="px-3"
          >
            Sonraki
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>
      )}
    </div>
  );
} 