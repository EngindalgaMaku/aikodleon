import { Metadata } from 'next';
import HomePageClientContent from '@/components/HomePageClientContent';

export const metadata: Metadata = {
  title: 'Kodleon | Türkiye\'nin Lider Yapay Zeka Eğitim Platformu',
  description: 'Kodleon ile yapay zeka dünyasındaki en son gelişmeleri öğrenin, geleceğin teknolojilerini şekillendiren AI becerileri kazanın ve kariyerinize yön verin. Uzman eğitmenler, kapsamlı içerik ve uygulamalı projelerle uzmanlaşın.',
  keywords: 'yapay zeka eğitimi, AI kursları, kodleon, makine öğrenmesi, doğal dil işleme, bilgisayarlı görü, derin öğrenme, Türkçe yapay zeka, online AI eğitimi, yapay zeka projeleri, AI sertifikası, veri bilimi eğitimi, yapay zeka uzmanlığı',
  alternates: {
    canonical: 'https://kodleon.com',
  },
  openGraph: {
    type: "website",
    locale: "tr_TR",
    url: "https://kodleon.com",
    title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
    description: "Kodleon ile yapay zeka dünyasındaki en son gelişmeleri öğrenin, geleceğin teknolojilerini şekillendiren AI becerileri kazanın ve kariyerinize yön verin.",
    images: [
      {
        url: "/images/og-image.png",
        width: 1200,
        height: 630,
        alt: "Kodleon Yapay Zeka Eğitim Platformu | Geleceği Kodlayın"
      }
    ],
  },
};

export default function Home() {
  return <HomePageClientContent />;
}