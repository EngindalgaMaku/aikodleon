'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Code2, BookOpen, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function OOPExamples() {
  const examples = [
    {
      title: "Basit Sınıf Örneği",
      description: "Temel bir sınıf oluşturma ve kullanma örneği",
      code: `class Araba:
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model
    
    def bilgi_goster(self):
        return f"{self.marka} {self.model}"

# Kullanım örneği
araba1 = Araba("Toyota", "Corolla")
print(araba1.bilgi_goster())  # Toyota Corolla`
    },
    {
      title: "Kalıtım Örneği",
      description: "Bir sınıftan başka bir sınıf türetme örneği",
      code: `class Hayvan:
    def __init__(self, isim):
        self.isim = isim
    
    def ses_cikar(self):
        pass

class Kopek(Hayvan):
    def ses_cikar(self):
        return f"{self.isim} havlıyor: Hav hav!"

# Kullanım örneği
kopek = Kopek("Karabaş")
print(kopek.ses_cikar())  # Karabaş havlıyor: Hav hav!`
    },
    {
      title: "Kapsülleme Örneği",
      description: "Private değişkenler ve metodlar kullanma örneği",
      code: `class BankaHesabi:
    def __init__(self, bakiye):
        self.__bakiye = bakiye  # private değişken
    
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            return f"{miktar} TL yatırıldı. Yeni bakiye: {self.__bakiye} TL"
        return "Geçersiz miktar"

# Kullanım örneği
hesap = BankaHesabi(1000)
print(hesap.para_yatir(500))  # 500 TL yatırıldı. Yeni bakiye: 1500 TL`
    }
  ];

  return (
    <div className="container max-w-4xl mx-auto py-8 px-4">
      <div className="mb-8">
        <Link 
          href="/topics/python/nesne-tabanli-programlama" 
          className="inline-flex items-center text-primary hover:underline mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Nesne Tabanlı Programlama'ya Dön
        </Link>
        <h1 className="text-3xl font-bold mb-2">Kod Örnekleri</h1>
        <p className="text-muted-foreground">
          Nesne tabanlı programlama konusunda öğrendiklerinizi pekiştirmek için hazırlanmış örnek kodlar.
        </p>
      </div>

      <div className="grid gap-6">
        {examples.map((example, index) => (
          <Card key={index} className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code2 className="h-5 w-5 text-primary" />
                {example.title}
              </CardTitle>
              <CardDescription>{example.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code className="text-sm">{example.code}</code>
              </pre>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
} 