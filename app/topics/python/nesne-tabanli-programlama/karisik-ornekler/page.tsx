'use client';

import { useState } from 'react';
import { examples } from './examples-data';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ChevronLeft, ChevronRight, ArrowLeft } from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import Link from 'next/link';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

const ITEMS_PER_PAGE = 4;

export default function MixedExamplesPage() {
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.ceil(examples.length / ITEMS_PER_PAGE);

  const goToNextPage = () => {
    setCurrentPage((prev) => (prev < totalPages ? prev + 1 : prev));
  };

  const goToPreviousPage = () => {
    setCurrentPage((prev) => (prev > 1 ? prev - 1 : prev));
  };
  
  const getDifficultyClass = (difficulty: 'Kolay' | 'Orta' | 'Zor') => {
    switch (difficulty) {
      case 'Kolay':
        return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300';
      case 'Orta':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300';
      case 'Zor':
        return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300';
    }
  };

  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const selectedExamples = examples.slice(startIndex, startIndex + ITEMS_PER_PAGE);

  return (
    <div className="container mx-auto px-4 py-8">
       <div className="max-w-4xl mx-auto">
        <Button variant="ghost" asChild className="mb-6">
          <Link href="/topics/python/nesne-tabanli-programlama" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Geri Dön
          </Link>
        </Button>
      
        <div className="text-center mb-10">
            <h1 className="text-4xl font-bold tracking-tight">Karışık OOP Örnekleri</h1>
            <p className="mt-2 text-lg text-muted-foreground">
            Python Nesne Tabanlı Programlama bilginizi test etmek ve pekiştirmek için çeşitli alıştırmalar.
            </p>
        </div>

        <div className="space-y-8">
            {selectedExamples.map((example) => (
            <Card key={example.id} className="overflow-hidden">
                <CardHeader>
                <div className="flex justify-between items-start">
                    <div className="flex-grow">
                    <CardTitle className="text-2xl mb-2">{example.id}. {example.title}</CardTitle>
                    <div className="flex gap-2 flex-wrap">
                        {example.topics.map((topic) => (
                        <Badge key={topic} variant="secondary">{topic}</Badge>
                        ))}
                    </div>
                    </div>
                    <Badge className={`text-sm ${getDifficultyClass(example.difficulty)}`}>{example.difficulty}</Badge>
                </div>
                </CardHeader>
                <CardContent>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                    <MarkdownContent content={example.description} />
                </div>
                </CardContent>
                <CardFooter>
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="solution">
                      <AccordionTrigger className="text-sm font-semibold">
                        Çözümü Görüntüle
                      </AccordionTrigger>
                      <AccordionContent>
                        <MarkdownContent content={example.solution} />
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardFooter>
            </Card>
            ))}
        </div>

        <div className="flex items-center justify-center space-x-4 mt-8">
            <Button
            variant="outline"
            size="icon"
            onClick={goToPreviousPage}
            disabled={currentPage === 1}
            >
            <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm font-medium">
            Sayfa {currentPage} / {totalPages}
            </span>
            <Button
            variant="outline"
            size="icon"
            onClick={goToNextPage}
            disabled={currentPage === totalPages}
            >
            <ChevronRight className="h-4 w-4" />
            </Button>
        </div>
      </div>
    </div>
  );
} 