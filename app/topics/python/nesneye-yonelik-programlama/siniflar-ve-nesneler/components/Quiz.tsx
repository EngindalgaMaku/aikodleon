'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface QuizQuestion {
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const quizQuestions: QuizQuestion[] = [
  {
    question: "Python'da yeni bir sınıf oluşturmak için hangi anahtar kelime kullanılır?",
    options: ["def", "class", "create", "new"],
    correctAnswer: 1,
    explanation: "Python'da yeni bir sınıf oluşturmak için 'class' anahtar kelimesi kullanılır."
  },
  {
    question: "Bir sınıfın yapıcı (constructor) metodu hangi isimle tanımlanır?",
    options: ["__init__", "__new__", "__create__", "__start__"],
    correctAnswer: 0,
    explanation: "Python'da constructor metodu '__init__' olarak tanımlanır ve nesne oluşturulduğunda otomatik olarak çağrılır."
  },
  {
    question: "Aşağıdakilerden hangisi instance variable (örnek değişkeni) tanımlamak için doğru sözdizimini gösterir?",
    options: ["self.variable", "this.variable", "variable", "@variable"],
    correctAnswer: 0,
    explanation: "Instance variable'lar 'self' anahtar kelimesi kullanılarak tanımlanır."
  },
  {
    question: "Bir sınıf metodunun ilk parametresi ne olmalıdır?",
    options: ["this", "self", "instance", "object"],
    correctAnswer: 1,
    explanation: "Python'da sınıf metodlarının ilk parametresi geleneksel olarak 'self' olmalıdır."
  },
  {
    question: "Aşağıdaki kod parçasının çıktısı ne olur?\\nclass Test:\\n    def __init__(self, x):\\n        self.x = x\\n\\nt = Test(5)\\nprint(t.x)",
    options: ["5", "x", "None", "Hata verir"],
    correctAnswer: 0,
    explanation: "Nesne oluşturulurken x parametresine 5 değeri atanır ve t.x ile bu değere erişilir."
  },
  {
    question: "Bir sınıftan nesne oluşturmak için doğru sözdizimi nedir?",
    options: ["object = new Class()", "object = Class.new()", "object = Class()", "object = create Class()"],
    correctAnswer: 2,
    explanation: "Python'da nesne oluşturmak için sınıf adı fonksiyon gibi çağrılır: Class()"
  },
  {
    question: "Instance metodları ile ilgili hangisi doğrudur?",
    options: [
      "Static metodlardır",
      "Nesnenin durumunu değiştirebilir",
      "self parametresi almazlar",
      "Sınıf dışından çağrılamazlar"
    ],
    correctAnswer: 1,
    explanation: "Instance metodları nesnenin durumunu değiştirebilir ve nesnenin özelliklerine erişebilir."
  },
  {
    question: "Aşağıdakilerden hangisi bir sınıfın özelliklerine (attributes) erişmek için kullanılır?",
    options: ["nesne->ozellik", "nesne::ozellik", "nesne.ozellik", "nesne@ozellik"],
    correctAnswer: 2,
    explanation: "Python'da nesne özelliklerine nokta (.) operatörü ile erişilir."
  },
  {
    question: "Constructor (__init__) metodu ne zaman çağrılır?",
    options: [
      "Nesne silindiğinde",
      "Nesne oluşturulduğunda",
      "İlk metod çağrıldığında",
      "Sınıf tanımlandığında"
    ],
    correctAnswer: 1,
    explanation: "Constructor metodu, sınıftan yeni bir nesne oluşturulduğunda otomatik olarak çağrılır."
  },
  {
    question: "Aşağıdaki kodun çıktısı ne olur?\\nclass A:\\n    def __init__(self):\\n        self.x = 1\\n\\na = A()\\nb = A()\\nb.x = 2\\nprint(a.x)",
    options: ["1", "2", "None", "Hata verir"],
    correctAnswer: 0,
    explanation: "Her nesne kendi instance variable'larına sahiptir. b.x'in değişmesi a.x'i etkilemez."
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
    
    if (answerIndex === quizQuestions[currentQuestionIndex].correctAnswer) {
      setScore(prev => prev + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < quizQuestions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    } else {
      setQuizCompleted(true);
    }
  };

  const getButtonVariant = (index: number): "default" | "destructive" | "outline" | "secondary" => {
    if (selectedAnswer === null) return "outline";
    if (selectedAnswer === index) {
      return index === quizQuestions[currentQuestionIndex].correctAnswer ? "secondary" : "destructive";
    }
    return index === quizQuestions[currentQuestionIndex].correctAnswer && showResult ? "secondary" : "outline";
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-2xl font-bold">Konu Sonu Quiz</CardTitle>
        <CardDescription>
          Sınıflar ve nesneler konusundaki bilgilerinizi test edin.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!quizStarted ? (
          <div className="text-center">
            <p className="mb-4">
              Bu quiz {quizQuestions.length} sorudan oluşmaktadır ve Python'da sınıflar ve nesneler konusunu kapsamaktadır.
            </p>
            <Button onClick={handleStartQuiz}>
              Quize Başla
            </Button>
          </div>
        ) : quizCompleted ? (
          <div className="text-center">
            <h3 className="text-xl font-bold mb-4">Quiz Tamamlandı!</h3>
            <p className="text-lg mb-4">
              Skorunuz: {score}/{quizQuestions.length} ({Math.round((score/quizQuestions.length)*100)}%)
            </p>
            <Button onClick={handleStartQuiz}>
              Quizi Tekrar Başlat
            </Button>
          </div>
        ) : (
          <div>
            <div className="mb-6">
              <p className="text-sm text-muted-foreground mb-2">
                Soru {currentQuestionIndex + 1}/{quizQuestions.length}
              </p>
              <div className="h-2 w-full bg-muted rounded-full">
                <div 
                  className="h-2 bg-primary rounded-full transition-all"
                  style={{ width: `${((currentQuestionIndex + 1) / quizQuestions.length) * 100}%` }}
                />
              </div>
            </div>

            <div className="mb-6">
              <p className="text-lg mb-4 whitespace-pre-wrap">
                {quizQuestions[currentQuestionIndex].question}
              </p>
              <div className="space-y-2">
                {quizQuestions[currentQuestionIndex].options.map((option, index) => (
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
                  selectedAnswer === quizQuestions[currentQuestionIndex].correctAnswer
                    ? "bg-green-100 text-green-800"
                    : "bg-red-100 text-red-800"
                }`}>
                  {quizQuestions[currentQuestionIndex].explanation}
                </p>
              </div>
            )}

            {showResult && (
              <Button onClick={handleNextQuestion}>
                {currentQuestionIndex < quizQuestions.length - 1 ? "Sonraki Soru" : "Quizi Tamamla"}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 