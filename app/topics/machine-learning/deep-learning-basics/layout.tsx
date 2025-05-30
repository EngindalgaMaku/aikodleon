/* THIS IS A NEW FILE */
import { Metadata } from 'next';

const pageTitle = "Derin Öğrenme Temelleri";
const pageDescription = "Derin öğrenmenin temel kavramlarını, yapay sinir ağlarının çalışma prensiplerini ve öğrenme süreçlerini keşfedin.";
const pageKeywords = "derin öğrenme, yapay sinir ağları, nöronlar, aktivasyon fonksiyonları, geri yayılım, derin öğrenme temelleri, kodleon, makine öğrenmesi";
const pageUrl = "https://kodleon.com/topics/machine-learning/deep-learning-basics";
const imageUrl = "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2";

export const metadata: Metadata = {
  title: `${pageTitle} | Kodleon`,
  description: pageDescription,
  keywords: pageKeywords,
  alternates: {
    canonical: pageUrl,
  },
  openGraph: {
    title: `${pageTitle} | Kodleon`,
    description: pageDescription,
    url: pageUrl,
    images: [
      {
        url: imageUrl,
        width: 1200,
        height: 630,
        alt: `${pageTitle} - Kodleon`,
      }
    ],
    type: 'article',
  },
  twitter: {
    card: 'summary_large_image',
    title: `${pageTitle} | Kodleon`,
    description: pageDescription,
    images: [imageUrl],
  },
};

export default function DeepLearningBasicsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
} 