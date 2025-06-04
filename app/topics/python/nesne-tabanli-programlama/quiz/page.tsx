'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface Question {
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const questions: Question[] = [
  {
    question: "Python'da sınıf tanımlamak için hangi anahtar kelime kullanılır?",
    options: ["def", "class", "function", "create"],
    correctAnswer: 1,
    explanation: "Python'da sınıf tanımlamak için 'class' anahtar kelimesi kullanılır."
  },
  {
    question: "Aşağıdakilerden hangisi bir constructor metodudur?",
    options: ["__init__", "__main__", "__str__", "__call__"],
    correctAnswer: 0,
    explanation: "'__init__' metodu, bir sınıfın constructor metodudur ve nesne oluşturulduğunda otomatik olarak çağrılır."
  },
  {
    question: "Bir sınıf metodunun ilk parametresi ne olmalıdır?",
    options: ["this", "self", "instance", "object"],
    correctAnswer: 1,
    explanation: "Python'da sınıf metodlarının ilk parametresi geleneksel olarak 'self' olmalıdır."
  },
  {
    question: "Aşağıdakilerden hangisi bir instance variable tanımlama yöntemidir?",
    options: ["variable = value", "self.variable = value", "@variable = value", "def variable = value"],
    correctAnswer: 1,
    explanation: "Instance variable'lar 'self' anahtar kelimesi kullanılarak tanımlanır: self.variable = value"
  },
  {
    question: "Bir sınıftan nesne oluşturmak için hangi sözdizimi kullanılır?",
    options: ["MyClass.new()", "new MyClass()", "MyClass()", "create MyClass()"],
    correctAnswer: 2,
    explanation: "Python'da bir sınıftan nesne oluşturmak için sınıf adı fonksiyon gibi çağrılır: MyClass()"
  }
];

export default function Quiz() {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [quizStarted, setQuizStarted] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);

  const handleStartQuiz = () => {
    setQuizStarted(true);
    setScore(0);
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setQuizCompleted(false);
  };

  const handleAnswerSelect = (answerIndex: number) => {
    if (showResult) return;
    
    setSelectedAnswer(answerIndex);
    setShowResult(true);
    
    if (answerIndex === questions[currentQuestionIndex].correctAnswer) {
      setScore(prev => prev + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    } else {
      setQuizCompleted(true);
    }
  };

  const getButtonVariant = (index: number) => {
    if (selectedAnswer === null) return "outline";
    if (selectedAnswer === index) {
      return index === questions[currentQuestionIndex].correctAnswer ? "secondary" : "destructive";
    }
    return index === questions[currentQuestionIndex].correctAnswer && showResult ? "secondary" : "outline";
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-2xl font-bold">Python OOP Quiz</CardTitle>
        <CardDescription>
          Nesne yönelimli programlama bilginizi test edin.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!quizStarted ? (
          <div className="text-center">
            <p className="mb-4">
              Bu quiz 5 sorudan oluşmaktadır ve Python'da nesne yönelimli programlama konularını kapsamaktadır.
            </p>
            <Button onClick={handleStartQuiz}>
              Quize Başla
            </Button>
          </div>
        ) : quizCompleted ? (
          <div className="text-center">
            <h3 className="text-xl font-bold mb-4">Quiz Tamamlandı!</h3>
            <p className="text-lg mb-4">
              Skorunuz: {score}/{questions.length} ({Math.round((score/questions.length)*100)}%)
            </p>
            <Button onClick={handleStartQuiz}>
              Quizi Tekrar Başlat
            </Button>
          </div>
        ) : (
          <div>
            <div className="mb-6">
              <p className="text-sm text-muted-foreground mb-2">
                Soru {currentQuestionIndex + 1}/{questions.length}
              </p>
              <div className="h-2 w-full bg-muted rounded-full">
                <div 
                  className="h-2 bg-primary rounded-full transition-all"
                  style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
                />
              </div>
            </div>

            <div className="mb-6">
              <p className="text-lg mb-4">
                {questions[currentQuestionIndex].question}
              </p>
              <div className="space-y-2">
                {questions[currentQuestionIndex].options.map((option, index) => (
                  <Button
                    key={index}
                    variant={getButtonVariant(index)}
                    className="w-full justify-start h-auto py-3 px-4"
                    onClick={() => handleAnswerSelect(index)}
                    disabled={showResult}
                  >
                    {option}
                  </Button>
                ))}
              </div>
            </div>

            {showResult && (
              <div className="mb-6">
                <p className={`p-4 rounded-lg ${
                  selectedAnswer === questions[currentQuestionIndex].correctAnswer
                    ? "bg-green-100 text-green-800"
                    : "bg-red-100 text-red-800"
                }`}>
                  {questions[currentQuestionIndex].explanation}
                </p>
              </div>
            )}

            {showResult && (
              <Button onClick={handleNextQuestion}>
                {currentQuestionIndex < questions.length - 1 ? "Sonraki Soru" : "Quizi Tamamla"}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 