import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Python OOP: Kalıtım (Inheritance) | Kodleon',
  description: 'Python\'da kalıtım kavramını, türetilmiş sınıfları ve çoklu kalıtımı öğrenin.',
};

export default function KalitimLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 