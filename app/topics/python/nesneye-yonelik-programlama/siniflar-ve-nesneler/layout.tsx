import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Python OOP: Sınıflar ve Nesneler | Kodleon',
  description: 'Python\'da sınıf ve nesne kavramlarını, oluşturma yöntemlerini ve kullanım örneklerini öğrenin.',
};

export default function Layout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 