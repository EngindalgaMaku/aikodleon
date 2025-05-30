import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Brain, BookOpen, Award, Users, Lightbulb, Rocket } from "lucide-react";

export const metadata: Metadata = createPageMetadata({
  title: 'Hakkımızda',
  description: 'Kodleon yapay zeka eğitim platformu hakkında bilgi edinin. Vizyonumuz, misyonumuz ve ekibimizle tanışın.',
  path: '/about',
  keywords: ['kodleon hakkında', 'yapay zeka eğitim platformu', 'misyon', 'vizyon', 'ekip', 'türkiye ai eğitimi'],
});

export default function AboutPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
      <h1 className="text-4xl font-bold mb-6">Hakkımızda</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Kodleon, yapay zeka alanında Türkçe kaynaklar sunarak Türkiye'deki bireylerin ve kurumların bu devrimci teknolojiyi öğrenmelerini ve kullanmalarını kolaylaştırmayı hedefleyen bir eğitim platformudur.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Vizyonumuz</h2>
        <p>Herkes için erişilebilir ve yüksek kaliteli yapay zeka eğitimi sunarak Türkiye'nin teknolojik dönüşümüne katkı sağlamak.</p>

        <h2>Misyonumuz</h2>
        <p>En güncel yapay zeka konularında pratik ve anlaşılır eğitim içerikleri geliştirmek, öğrenme topluluğunu desteklemek ve yapay zeka okuryazarlığını artırmak.</p>

        <h2>Ekibimiz</h2>
        <p>Kodleon, yapay zeka, yazılım geliştirme ve eğitim alanlarında tutkulu uzmanlardan oluşan bir ekip tarafından yönetilmektedir. Ekibimiz, kaliteli eğitim materyalleri sunmak ve öğrenme deneyiminizi en üst düzeye çıkarmak için çalışmaktadır.</p>
        
        <h3>Neden Kodleon?</h3>
        <ul>
          <li>**Güncel İçerikler:** Sürekli güncellenen ve sektördeki en son trendleri takip eden eğitimler.</li>
          <li>**Uygulamalı Yaklaşım:** Teorik bilginin yanı sıra pratik projelerle öğrenmeyi pekiştirme.</li>
          <li>**Türkçe Kaynaklar:** Yapay zeka eğitimine anadilinizde erişim kolaylığı.</li>
          <li>**Erişilebilirlik:** Farklı seviyelerdeki kullanıcılar için uygun içerikler.</li>
        </ul>

        <p>Yapay zeka yolculuğunuzda size eşlik etmekten heyecan duyuyoruz.</p>

        </div>
    </div>
  );
}