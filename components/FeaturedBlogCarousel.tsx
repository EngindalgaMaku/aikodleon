'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { ArrowRight, ChevronLeft, ChevronRight } from 'lucide-react';

interface BlogPostData {
  title: string;
  snippet: string;
  imageUrl: string;
  href: string;
  date?: string; 
  category: string;
}

const formatDate = (dateInput: string | Date): string => {
  const date = typeof dateInput === 'string' ? new Date(dateInput) : dateInput;
  return date.toLocaleDateString('tr-TR', { year: 'numeric', month: 'long', day: 'numeric' });
};

interface FeaturedBlogCarouselProps {
  posts: BlogPostData[];
}

const FeaturedBlogCarousel: React.FC<FeaturedBlogCarouselProps> = ({ posts }) => {
  const postsToShow = posts.slice(0, 3);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  const handleNext = useCallback(() => {
    setIsTransitioning(true);
    setTimeout(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % postsToShow.length);
      setIsTransitioning(false);
    }, 300); 
  }, [postsToShow.length]);

  const handlePrev = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setCurrentIndex((prevIndex) => (prevIndex - 1 + postsToShow.length) % postsToShow.length);
      setIsTransitioning(false);
    }, 300); 
  };

  useEffect(() => {
    if (postsToShow.length <= 1) return;
    const interval = setInterval(() => {
      handleNext();
    }, 7000);
    return () => clearInterval(interval);
  }, [handleNext, postsToShow.length]);

  if (postsToShow.length === 0) {
    return null; 
  }

  const activePost = postsToShow[currentIndex];

  return (
    <div className="max-w-4xl mx-auto relative h-[450px] md:h-[500px] z-20 isolate">
      <div className="relative rounded-2xl overflow-hidden shadow-2xl hover:shadow-3xl transition-shadow duration-300 group h-full">
        <div className={`relative h-full w-full transition-opacity duration-300 ease-in-out ${isTransitioning ? 'opacity-50' : 'opacity-100'}`}>
          <Image
            src={activePost.imageUrl}
            alt={activePost.title}
            fill
            className="object-cover pointer-events-none"
            priority={currentIndex === 0} 
            sizes="(max-width: 768px) 100vw, (max-width: 1024px) 80vw, 1000px"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent pointer-events-none z-0" />
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-black/40 backdrop-blur-sm rounded-lg p-4 md:p-6 w-full max-w-lg shadow-lg border border-white/10 pointer-events-auto z-10">
            <div className="flex items-center gap-3 mb-4">
              <span className="px-4 py-1.5 rounded-full text-sm font-semibold bg-primary/90 text-primary-foreground shadow-sm">
                {activePost.category}
              </span>
              {activePost.date && (
                <span className="text-sm text-gray-200 font-medium">{formatDate(activePost.date)}</span>
              )}
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-3 leading-tight drop-shadow">
              {activePost.title}
            </h2>
            <p className="text-gray-100 mb-5 line-clamp-2 md:line-clamp-3 text-base md:text-lg font-normal drop-shadow">
              {activePost.snippet}
            </p>
            <Button asChild variant="default" size="sm" className="rounded-full shadow hover:shadow-md transition-all font-semibold text-base px-5 py-2.5">
              <Link href={activePost.href}>
                Devamını Oku
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
        <Button
          variant="outline"
          size="icon"
          className="absolute left-3 top-1/2 -translate-y-1/2 z-10 bg-background/70 hover:bg-background/90 text-foreground rounded-full shadow-md hover:scale-105 transition-all opacity-0 group-hover:opacity-100 pointer-events-auto"
          onClick={handlePrev}
          aria-label="Önceki Yazı"
        >
          <ChevronLeft className="h-6 w-6" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="absolute right-3 top-1/2 -translate-y-1/2 z-10 bg-background/70 hover:bg-background/90 text-foreground rounded-full shadow-md hover:scale-105 transition-all opacity-0 group-hover:opacity-100 pointer-events-auto"
          onClick={handleNext}
          aria-label="Sonraki Yazı"
        >
          <ChevronRight className="h-6 w-6" />
        </Button>
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex space-x-2 pointer-events-auto">
          {postsToShow.map((_, index) => (
            <button
              key={index}
              onClick={() => {
                setIsTransitioning(true);
                setTimeout(() => {
                  setCurrentIndex(index);
                  setIsTransitioning(false);
                }, 300);
              }}
              aria-label={`Yazı ${index + 1}`}
              className={`h-2.5 w-2.5 rounded-full transition-all duration-300 ease-in-out ${currentIndex === index ? 'bg-primary scale-125' : 'bg-gray-400 hover:bg-gray-200'}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default FeaturedBlogCarousel; 