import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'DevOps Practices | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için DevOps pratikleri. CI/CD, infrastructure as code, monitoring ve automation.',
};

export default function DevOpsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 