import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Cloud Deployment | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için cloud deployment. Containerization, cloud platforms, CI/CD ve infrastructure as code.',
};

export default function DeploymentLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 