'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, CheckCircle2, XCircle } from "lucide-react";
import Link from "next/link";

export default function OOPQuiz() {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);

  const questions = [
    {
      question: "Nesne tabanlı programlamada bir sınıftan nesne oluşturma işlemine ne ad verilir?",
      options: [
        "Kalıtım",
        "Örnekleme (Instantiation)",
        "Kapsülleme",
        "Soyutlama"
      ],
      correctAnswer: 1
    },
    {
      question: "Aşağıdakilerden hangisi Python'da bir sınıf oluştururken kullanılan anahtar kelimedir?",
      options: [
        "def",
        "function",
        "class",
        "create"
      ],
      correctAnswer: 2
    },
    {
      question: "Bir sınıfın yapıcı (constructor) metodu Python'da hangi isimle tanımlanır?",
      options: [
        "__init__",
        "constructor",
        "build",
        "create"
      ],
      correctAnswer: 0
    },
    {
      question: "Bir sınıfın özelliklerini ve davranışlarını başka bir sınıfa aktarma işlemine ne ad verilir?",
      options: [
        "Kapsülleme",
        "Çok Biçimlilik",
        "Kalıtım",
        "Soyutlama"
      ],
      correctAnswer: 2
    },
    {
      question: "Python'da private bir değişken oluştururken değişken isminin başına hangi karakter(ler) getirilir?",
      options: [
        "_",
        "__",
        "#",
        "private"
      ],
      correctAnswer: 1
    }
  ];

  const handleAnswerSelect = (answerIndex: number) => {
    setSelectedAnswer(answerIndex);
    setShowResult(true);
    
    if (answerIndex === questions[currentQuestionIndex].correctAnswer) {
      setScore(score + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    }
  };

  const handleRestartQuiz = () => {
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
  };

  const currentQuestion = questions[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === questions.length - 1;

  return (
    <div className="container max-w-3xl mx-auto py-8 px-4">
      <div className="mb-8">
        <Link 
          href="/topics/python/nesne-tabanli-programlama" 
          className="inline-flex items-center text-primary hover:underline mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Nesne Tabanlı Programlama'ya Dön
        </Link>
        <h1 className="text-3xl font-bold mb-2">Quiz Soruları</h1>
        <p className="text-muted-foreground">
          Nesne tabanlı programlama konusundaki bilgilerinizi test edin.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Soru {currentQuestionIndex + 1}/{questions.length}</CardTitle>
          <CardDescription>
            Doğru cevaplar: {score}/{currentQuestionIndex + (showResult ? 1 : 0)}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-lg mb-4">{currentQuestion.question}</p>
          <div className="grid gap-3">
            {currentQuestion.options.map((option, index) => (
              <Button
                key={index}
                variant={selectedAnswer === null ? "outline" : 
                  selectedAnswer === index ? 
                    (index === currentQuestion.correctAnswer ? "success" : "destructive") :
                    index === currentQuestion.correctAnswer && showResult ? "success" : "outline"}
                className="justify-start h-auto py-3 px-4"
                onClick={() => !showResult && handleAnswerSelect(index)}
                disabled={showResult}
              >
                {showResult && index === currentQuestion.correctAnswer && (
                  <CheckCircle2 className="mr-2 h-4 w-4 text-green-500" />
                )}
                {showResult && index === selectedAnswer && index !== currentQuestion.correctAnswer && (
                  <XCircle className="mr-2 h-4 w-4 text-red-500" />
                )}
                {option}
              </Button>
            ))}
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          {showResult && !isLastQuestion && (
            <Button onClick={handleNextQuestion}>
              Sonraki Soru
            </Button>
          )}
          {isLastQuestion && showResult && (
            <Button onClick={handleRestartQuiz}>
              Quizi Yeniden Başlat
            </Button>
          )}
        </CardFooter>
      </Card>
    </div>
  );
} 