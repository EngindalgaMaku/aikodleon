'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Lightbulb, ChevronDown, ChevronRight, ChevronLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import MarkdownContent from '@/components/MarkdownContent';
import { examples, Example } from './examples-data';

const EXAMPLES_PER_PAGE = 4;

export default function MixedExamplesPage() {
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.ceil(examples.length / EXAMPLES_PER_PAGE);
  const startIndex = (currentPage - 1) * EXAMPLES_PER_PAGE;
  const selectedExamples = examples.slice(startIndex, startIndex + EXAMPLES_PER_PAGE);

  const handlePageChange = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
      window.scrollTo(0, 0);
    }
  };

  const getDifficultyClass = (difficulty: Example['difficulty']) => {
    switch (difficulty) {
      case 'Kolay':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/50 dark:text-green-200 dark:border-green-700';
      case 'Orta':
        return 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/50 dark:text-amber-200 dark:border-amber-700';
      case 'Zor':
        return 'bg-rose-100 text-rose-800 border-rose-200 dark:bg-rose-900/50 dark:text-rose-200 dark:border-rose-700';
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <Button variant="ghost" asChild>
            <Link href="/topics/python/temel-python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python Temelleri
            </Link>
          </Button>
        </div>

        <div className="mb-8 text-center">
          <Lightbulb className="mx-auto h-12 w-12 text-amber-500 mb-4" />
          <h1 className="text-4xl font-bold tracking-tight">Karışık Örnekler</h1>
          <p className="mt-2 text-lg text-muted-foreground">
            Öğrendiğiniz konuları birleştirerek pratik yapın ve bilginizi test edin.
          </p>
        </div>

        <div className="space-y-8">
          {selectedExamples.map((example) => (
            <Card key={example.id}>
              <CardHeader>
                <div className="flex justify-between items-start">
                  <CardTitle>{example.id}. {example.title}</CardTitle>
                  <Badge className={getDifficultyClass(example.difficulty)}>
                    {example.difficulty}
                  </Badge>
                </div>
                <div className="pt-2">
                  {example.topics.map((topic) => (
                    <Badge key={topic} variant="secondary" className="mr-2 mb-2">
                      {topic}
                    </Badge>
                  ))}
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

        {/* Pagination */}
        <div className="mt-12 flex justify-center items-center gap-4">
          <Button
            variant="outline"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4 mr-2" />
            Önceki
          </Button>
          <span className="text-sm font-medium">
            Sayfa {currentPage} / {totalPages}
          </span>
          <Button
            variant="outline"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            Sonraki
            <ChevronRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      </div>
    </div>
  );
} 