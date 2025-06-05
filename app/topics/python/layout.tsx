import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Python Eğitimleri | Kodleon',
  description: 'Python programlama dilini temellerden ileri seviyeye kadar öğrenin. Veri yapıları, algoritmalar, web geliştirme, veri bilimi ve daha fazlası.',
};

export default function PythonLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 