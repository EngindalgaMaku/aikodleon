import Link from "next/link";
import Image from "next/image";
import { ArrowRight, BookOpen, FileText, Video, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

const resources = [
  {
    title: "AI Temelleri E-Kitabı",
    description: "Yapay zeka temellerini kapsamlı bir şekilde anlatan e-kitap.",
    type: "E-Kitap",
    icon: <BookOpen className="h-6 w-6" />,
    imageUrl: "https://images.pexels.com/photos/1181671/pexels-photo-1181671.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Python ile ML Uygulamaları",
    description: "Makine öğrenmesi algoritmalarının Python ile uygulamalı örnekleri.",
    type: "Uygulama",
    icon: <FileText className="h-6 w-6" />,
    imageUrl: "https://images.pexels.com/photos/1181677/pexels-photo-1181677.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Derin Öğrenme Video Serisi",
    description: "Derin öğrenme konularını adım adım anlatan video eğitim serisi.",
    type: "Video",
    icon: <Video className="h-6 w-6" />,
    imageUrl: "https://images.pexels.com/photos/1181675/pexels-photo-1181675.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "NLP Araç Seti",
    description: "Doğal dil işleme projeleri için hazır kod ve araçlar.",
    type: "Araç Seti",
    icon: <Download className="h-6 w-6" />,
    imageUrl: "https://images.pexels.com/photos/1181673/pexels-photo-1181673.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  }
];

export default function ResourcesPage() {
  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Eğitim Kaynakları</h1>
        <p className="text-xl text-muted-foreground">
          Yapay zeka öğreniminizi destekleyecek kapsamlı eğitim materyalleri.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {resources.map((resource, index) => (
          <Card key={index} className="overflow-hidden">
            <div className="relative h-48">
              <Image 
                src={resource.imageUrl}
                alt={resource.title}
                fill
                className="object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
              <div className="absolute bottom-4 left-4 p-2 rounded-full bg-background/80 backdrop-blur-sm">
                {resource.icon}
              </div>
            </div>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{resource.title}</CardTitle>
                <span className="text-sm text-muted-foreground">{resource.type}</span>
              </div>
              <CardDescription>{resource.description}</CardDescription>
            </CardHeader>
            <CardFooter>
              <Button asChild variant="ghost" className="gap-1 ml-auto">
                <Link href="#">
                  İncele
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}