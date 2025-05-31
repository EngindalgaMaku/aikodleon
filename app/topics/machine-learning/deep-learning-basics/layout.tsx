/* THIS IS A NEW FILE */
import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';

const pageTitle = "Derin Öğrenme Temelleri";
const pageDescription = "Yapay sinir ağlarının katmanlı yapısını, temel mimarilerini (CNN, RNN) ve derin öğrenmenin makine öğrenmesindeki devrimsel etkilerini keşfedin.";
const pageKeywords = ["derin öğrenme", "deep learning", "yapay sinir ağları", "neural networks", "cnn", "rnn", "geri yayılım", "aktivasyon fonksiyonları", "makine öğrenmesi", "kodleon"];
const path = '/topics/machine-learning/deep-learning-basics';

export const metadata: Metadata = createPageMetadata({
  title: pageTitle,
  description: pageDescription,
  keywords: pageKeywords,
  path: path,
});

export default function DeepLearningBasicsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
} 