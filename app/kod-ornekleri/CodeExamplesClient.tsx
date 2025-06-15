'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

// Bu arayüzü ana sayfadan tekrar import etmek yerine burada tanımlıyoruz.
interface CodeExample {
  id: string;
  title: string;
  description: string;
  category: string;
  level: 'Başlangıç' | 'Orta' | 'İleri';
  image: string;
}

interface CodeExamplesClientProps {
  codeExamples: CodeExample[];
}

export default function CodeExamplesClient({ codeExamples }: CodeExamplesClientProps) {
  const [selectedCategory, setSelectedCategory] = useState('Tümü');

  const allCategories = ['Tümü', ...Array.from(new Set(codeExamples.map(ex => ex.category)))];

  const filteredExamples = selectedCategory === 'Tümü'
    ? codeExamples
    : codeExamples.filter(example => example.category === selectedCategory);

  return (
    <>
      {/* Filtreler */}
      <div className="flex flex-wrap gap-2 mb-8 justify-center">
        {allCategories.map(category => (
          <Button
            key={category}
            variant={selectedCategory === category ? 'default' : 'outline'}
            className="rounded-full"
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </Button>
        ))}
      </div>

      {/* Kod Örnekleri Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredExamples.map((example) => (
          <Card key={example.id} className="overflow-hidden flex flex-col h-full group">
            <div className="relative h-48">
              <Image
                src={example.image}
                alt={example.title}
                fill
                className="object-cover transition-transform duration-300 ease-in-out group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-black/10" />
              <div className="absolute top-4 left-4 flex gap-2">
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-black/40 text-white backdrop-blur-sm border border-white/20">
                  {example.category}
                </span>
                <span className={`px-3 py-1 rounded-full text-xs font-medium text-white backdrop-blur-sm border border-white/20 ${
                  example.level === 'Başlangıç' ? 'bg-green-600/80' :
                  example.level === 'Orta' ? 'bg-yellow-600/80' :
                  'bg-red-600/80'
                }`}>
                  {example.level}
                </span>
              </div>
            </div>
            <CardHeader>
              <CardTitle>{example.title}</CardTitle>
              <CardDescription>{example.description}</CardDescription>
            </CardHeader>
            <CardContent className="flex-grow">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>Python</span>
                <span>•</span>
                <span>Jupyter Notebook</span>
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild variant="default" className="w-full">
                <Link href={`/kod-ornekleri/${example.id}`}>
                  Kodu İncele
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </>
  );
} 