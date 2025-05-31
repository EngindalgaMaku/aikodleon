'use client';

import { useTranslation } from '@/lib/i18n';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Rss } from 'lucide-react';

export function TranslatedContent() {
  const { t } = useTranslation();

  return (
    <>
      <h1 
        id="hero-heading" 
        className="text-4xl md:text-6xl font-bold tracking-tight mb-6 
                   bg-clip-text text-transparent bg-gradient-to-r from-primary via-pink-500 to-orange-500 
                   animate-gradient-xy"
      >
        {t('home.hero.title')}
      </h1>
      <p className="text-xl md:text-2xl text-muted-foreground mb-10">
        {t('home.hero.subtitle')}
      </p>
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Button asChild size="lg" className="rounded-full shadow-lg hover:shadow-xl transition-shadow">
          <Link href="/topics">
            {t('navigation.topics.title')}
          </Link>
        </Button>
        <Button asChild size="lg" variant="outline" className="rounded-full border-border hover:border-primary/70 transition-colors">
          <Link href="/blog">
            <Rss className="mr-2 h-5 w-5" />
            {t('navigation.blog.title')}
          </Link>
        </Button>
      </div>
    </>
  );
} 