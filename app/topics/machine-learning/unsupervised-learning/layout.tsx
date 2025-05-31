/* THIS IS A NEW FILE */
import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';

const pageTitle = "Denetimsiz Öğrenme";
const pageDescription = "Etiketlenmemiş verilerden desenleri, yapıları ve ilişkileri bağımsız olarak keşfeden makine öğrenmesi dalını derinlemesine inceleyin.";
const pageKeywords = ["denetimsiz öğrenme", "kümeleme", "boyut indirgeme", "k-means", "pca", "dbscan", "apriori", "unsupervised learning", "makine öğrenmesi", "yapay zeka", "kodleon", "türkçe ai eğitimi"];
const path = '/topics/machine-learning/unsupervised-learning';

export const metadata: Metadata = createPageMetadata({
  title: pageTitle,
  description: pageDescription,
  keywords: pageKeywords,
  path: path,
});

export default function UnsupervisedLearningLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
} 