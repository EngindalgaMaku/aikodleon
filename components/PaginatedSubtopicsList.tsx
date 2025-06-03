"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import TopicSubtopicsList from "@/components/TopicSubtopicsList"; // Assuming this component can render a list of subtopics
import { ArrowLeft, ArrowRight } from "lucide-react";

interface Subtopic {
  title: string;
  description: string;
  imageUrl: string;
  href: string;
}

interface PaginatedSubtopicsListProps {
  subtopics: Subtopic[];
  isMetasearchTopic: boolean; // Keep this prop if TopicSubtopicsList uses it
  subtopicIcons: Record<string, JSX.Element>; // Keep this prop if TopicSubtopicsList uses it
  itemsPerPage?: number;
}

export default function PaginatedSubtopicsList({ 
  subtopics, 
  isMetasearchTopic, 
  subtopicIcons,
  itemsPerPage = 8 
}: PaginatedSubtopicsListProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = Math.ceil(subtopics.length / itemsPerPage);

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentSubtopics = subtopics.slice(startIndex, endIndex);

  const handlePreviousPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  };

  return (
    <div>
      <TopicSubtopicsList
        subtopics={currentSubtopics}
        isMetasearchTopic={false}
        subtopicIcons={subtopicIcons}
      />
      {totalPages > 1 && (
        <div className="flex justify-center items-center space-x-4 mt-8">
          <Button
            onClick={handlePreviousPage}
            disabled={currentPage === 1}
            variant="outline"
            size="sm"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Önceki
          </Button>
          <span className="text-sm text-muted-foreground">
            Sayfa {currentPage} / {totalPages}
          </span>
          <Button
            onClick={handleNextPage}
            disabled={currentPage === totalPages}
            variant="outline"
            size="sm"
          >
            Sonraki
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      )}
    </div>
  );
} 