'use client';

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import { ArrowLeft, ArrowRight, Shapes } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface Subtopic {
  title: string;
  description: string;
  imageUrl?: string;
  href: string;
}

interface TopicSubtopicsListProps {
  subtopics: Subtopic[];
  isMetasearchTopic: boolean; // To conditionally apply pagination
  subtopicIcons?: Record<string, JSX.Element>; // Optional: if you have specific icons for subtopics
}

const ITEMS_PER_PAGE = 4;

export default function TopicSubtopicsList({ subtopics, isMetasearchTopic, subtopicIcons }: TopicSubtopicsListProps) {
  const [currentPage, setCurrentPage] = useState(1);

  if (!subtopics || subtopics.length === 0) {
    return null; // Or some placeholder if no subtopics
  }

  const totalPages = isMetasearchTopic ? Math.ceil(subtopics.length / ITEMS_PER_PAGE) : 1;
  const currentSubtopicsToDisplay = isMetasearchTopic 
    ? subtopics.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)
    : subtopics;

  const handleNextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  };

  const handlePrevPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  return (
    <>
      <div className="space-y-4">
        {currentSubtopicsToDisplay.map((subtopic, index) => (
          <Link href={subtopic.href || '#'} key={index} className="block no-underline">
            <Card className="overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
              {subtopic.imageUrl && (
                <div className="relative h-40">
                  <Image 
                    src={subtopic.imageUrl}
                    alt={`${subtopic.title} alt konusu`}
                    fill
                    className="object-cover"
                    loading={index < 2 && currentPage === 1 ? "eager" : "lazy"} // Eager load first few on first page
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                </div>
              )}
              <CardHeader className="flex flex-row items-center gap-2">
                {subtopicIcons && subtopicIcons[subtopic.title] ? subtopicIcons[subtopic.title] : <Shapes className="h-5 w-5 text-primary" />}
                <CardTitle className="text-lg font-bold text-primary">{subtopic.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>{subtopic.description}</CardDescription>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {isMetasearchTopic && totalPages > 1 && (
        <div className="mt-8 flex justify-center items-center space-x-4">
          <Button onClick={handlePrevPage} disabled={currentPage === 1} variant="outline">
            <ArrowLeft className="mr-2 h-4 w-4" /> Önceki
          </Button>
          <span className="text-sm text-muted-foreground">
            Sayfa {currentPage} / {totalPages}
          </span>
          <Button onClick={handleNextPage} disabled={currentPage === totalPages} variant="outline">
            Sonraki <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      )}
    </>
  );
} 